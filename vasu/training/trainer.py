"""General-purpose VASU trainer with exact deterministic mid-epoch resume."""

from __future__ import annotations

import math
from pathlib import Path
import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vasu.training.accumulation import normalize_partial_accumulation
from vasu.training.checkpoint import save_checkpoint
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer
from vasu.training.resumable_sampler import ResumableBatchSampler
from vasu.training.resume_state import (
    FORMAT_VERSION,
    RESUME_PHASES,
    build_training_progress,
    capture_gradient_state,
    finite_gradients,
    restore_gradient_state,
    restore_rng_state,
    validate_gradient_presence,
)


class Trainer:
    """Train VASU models with additive exact-resume checkpoint metadata.

    Exact continuation is guaranteed for deterministic datasets and model
    execution.  With worker-side random transforms, callers must make those
    transforms stateless/deterministic; the sampler still restores the exact
    sample order even when DataLoader workers prefetch batches.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        train_dataset: Any,
        val_dataset: Any,
        config: Any,
        device: torch.device,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.config.use_amp,
        )
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 0
        self._accumulated_microbatches = 0
        self._optimizer_steps_in_epoch = 0
        self._resume_phase = "train"
        self.exact_resume_available = True
        self._build_dataloaders()
        self._load_checkpoint()

    def _build_dataloaders(self) -> None:
        workers = int(getattr(self.config, "num_workers", 0))
        persistent_workers = bool(
            getattr(self.config, "persistent_workers", False)
        )
        if persistent_workers:
            raise ValueError(
                "persistent_workers=True is not supported by exact resume; use "
                "stateless dataset transforms with persistent_workers=False."
            )
        self.train_sampler = ResumableBatchSampler(
            len(self.train_dataset),
            self.config.batch_size,
            shuffle=True,
            seed=int(getattr(self.config, "seed", 42)),
            drop_last=bool(getattr(self.config, "drop_last", True)),
        )
        if self.train_sampler.batches_per_epoch == 0:
            raise ValueError(
                "Training dataset produces zero batches for the configured "
                "batch_size and drop_last setting."
            )
        self.loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            pin_memory=True,
            num_workers=workers,
            persistent_workers=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=workers,
            persistent_workers=False,
        )

    def _load_checkpoint(self) -> None:
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            return
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            if self.scheduler.T_max != self.config.epochs:
                warnings.warn(
                    "Configured epochs differs from the checkpoint scheduler "
                    "horizon; continuation retains checkpoint "
                    f"T_max={self.scheduler.T_max}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.global_step = int(checkpoint.get("global_step", 0))

        progress = checkpoint.get("training_progress")
        if progress is None or progress.get("format_version") != FORMAT_VERSION:
            self.exact_resume_available = False
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            reason = (
                "has no training_progress sampler state"
                if progress is None
                else "uses an older training_progress format"
            )
            warnings.warn(
                f"Checkpoint {reason}. Falling back "
                "to legacy next-epoch resume; exact mid-epoch resume is "
                "unavailable and a historical mid-epoch checkpoint may skip its "
                "unprocessed remaining samples.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        phase = progress.get("phase")
        if phase not in RESUME_PHASES:
            raise ValueError(
                "Checkpoint training_progress has no unambiguous resume phase; "
                "exact resume is unsafe."
            )
        self.train_sampler.load_state_dict(progress["sampler"])
        self._resume_phase = phase
        self._accumulated_microbatches = int(
            progress["accumulated_microbatches"]
        )
        optimizer_steps = progress.get("optimizer_steps_in_epoch")
        if not isinstance(optimizer_steps, int) or optimizer_steps < 0:
            raise ValueError("Checkpoint optimizer_steps_in_epoch is invalid.")
        self._optimizer_steps_in_epoch = optimizer_steps
        if not 0 <= self._accumulated_microbatches < self.config.gradient_accumulation_steps:
            raise ValueError("Checkpoint accumulation position is invalid.")
        gradients = progress.get("gradients", {})
        validate_gradient_presence(gradients, self._accumulated_microbatches)
        restore_gradient_state(self.model, gradients)
        if "scaler" in progress:
            self.scaler.load_state_dict(progress["scaler"])
        restore_rng_state(progress["rng"])
        self.start_epoch = self.train_sampler.position.epoch
        if self._resume_phase == "next_epoch":
            # The checkpoint is already positioned at the first batch of the
            # next epoch, so its active runtime phase is ordinary training.
            self._resume_phase = "train"
        print(
            "Exact resume: "
            f"epoch={self.start_epoch}, "
            f"next_batch={self.train_sampler.position.next_batch_index}, "
            f"phase={self._resume_phase}, "
            f"global_step={self.global_step}."
        )

    def _training_progress(self) -> dict[str, Any]:
        return build_training_progress(
            sampler_state=self.train_sampler.state_dict(),
            phase=self._resume_phase,
            accumulated_microbatches=self._accumulated_microbatches,
            optimizer_steps_in_epoch=self._optimizer_steps_in_epoch,
            gradients=capture_gradient_state(self.model),
            scaler_state=self.scaler.state_dict(),
        )

    def save_training_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int | None = None,
        loss: float | None = None,
    ) -> None:
        """Save an additive exact-resume checkpoint at any microbatch boundary."""

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.start_epoch if epoch is None else epoch,
            loss=loss,
            path=path,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            training_progress=self._training_progress(),
        )

    def _unpack_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(batch) == 3:
            inputs, targets, mask = batch
            return inputs, targets, mask.to(self.device)
        inputs, targets = batch
        return inputs, targets, None

    def _optimizer_step(self) -> bool:
        """Apply one safe optimizer update and report whether it succeeded."""

        self.scaler.unscale_(self.optimizer)
        if not finite_gradients(self.model.parameters()):
            # GradScaler would skip this update for non-finite gradients. Make
            # that branch explicit so scheduler/global-step state cannot claim
            # an optimizer update that never happened.
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulated_microbatches = 0
            warnings.warn(
                "Skipped optimizer update because gradients were non-finite.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        normalize_partial_accumulation(
            self.model.parameters(),
            accumulated_microbatches=self._accumulated_microbatches,
            target_microbatches=self.config.gradient_accumulation_steps,
        )
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulated_microbatches = 0
        self.global_step += 1
        self._optimizer_steps_in_epoch += 1
        return True

    def train_epoch(self, *, max_microbatches: int | None = None) -> float:
        """Train remaining batches in the current epoch.

        ``max_microbatches`` is intentionally useful for controlled tests and
        interruptions.  A caller may save with :meth:`save_training_checkpoint`
        immediately afterward, including midway through gradient accumulation.
        """

        if max_microbatches is not None and max_microbatches <= 0:
            raise ValueError("max_microbatches must be positive when provided.")
        if self._resume_phase != "train":
            raise RuntimeError(
                "train_epoch is only valid during the train phase; current phase "
                f"is {self._resume_phase!r}."
            )
        self.model.train()
        total_loss = 0.0
        processed = 0
        progress = tqdm(self.loader, desc="Training", leave=False)
        if self._accumulated_microbatches == 0:
            self.optimizer.zero_grad(set_to_none=True)

        for batch in progress:
            inputs, targets, mask = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with torch.amp.autocast(self.device.type, enabled=self.config.use_amp):
                logits = self.model(inputs)
                raw_loss = language_model_loss(logits, targets, mask)
                scaled_loss = raw_loss / self.config.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            self.train_sampler.mark_batch_consumed()
            self._accumulated_microbatches += 1
            processed += 1
            total_loss += raw_loss.item()

            is_epoch_tail = self.train_sampler.epoch_complete
            if is_epoch_tail:
                self._resume_phase = "post_train_pre_validation"
            if (
                self._accumulated_microbatches
                == self.config.gradient_accumulation_steps
                or is_epoch_tail
            ):
                update_succeeded = self._optimizer_step()
                save_every = int(getattr(self.config, "save_every_steps", 0))
                if (
                    update_succeeded
                    and save_every > 0
                    and self.global_step % save_every == 0
                ):
                    self.save_training_checkpoint(
                        Path(self.config.checkpoint_dir)
                        / f"step_{self.global_step}.pt",
                        loss=raw_loss.item(),
                    )
            progress.set_postfix(loss=f"{raw_loss.item():.4f}")
            if max_microbatches is not None and processed >= max_microbatches:
                break
        progress.close()
        if processed == 0:
            raise RuntimeError("No training batches remain in the current epoch.")
        return total_loss / processed

    @torch.no_grad()
    def validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            inputs, targets, mask = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with torch.amp.autocast(self.device.type, enabled=self.config.use_amp):
                loss = language_model_loss(self.model(inputs), targets, mask)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self) -> None:
        if self.start_epoch >= self.config.epochs:
            print("Training already completed. Increase TrainConfig.epochs to continue.")
            return
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        for callback in self.callbacks:
            callback.on_train_begin(self)
        for epoch in range(self.start_epoch, self.config.epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            if self._resume_phase == "post_train_pre_validation":
                if not self.train_sampler.epoch_complete:
                    raise RuntimeError(
                        "post_train_pre_validation requires an exhausted sampler."
                    )
                train_loss = float("nan")
            elif self._resume_phase == "train":
                train_loss = self.train_epoch()
            elif self._resume_phase == "next_epoch":
                raise RuntimeError(
                    "next_epoch phase must use the sampler's next epoch before "
                    "entering fit."
                )
            else:  # Defensive guard for future checkpoint formats.
                raise RuntimeError(f"Unsupported resume phase: {self._resume_phase!r}")
            if not self.train_sampler.epoch_complete:
                raise RuntimeError("Epoch training ended before the sampler was exhausted.")
            val_loss = self.validate_epoch()
            if not math.isfinite(val_loss):
                raise FloatingPointError("Validation loss is non-finite; refusing checkpoint promotion.")
            if self._optimizer_steps_in_epoch > 0:
                self.scheduler.step()
            self.train_sampler.advance_epoch()
            self.start_epoch = self.train_sampler.position.epoch
            self._resume_phase = "next_epoch"
            self._optimizer_steps_in_epoch = 0
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_training_checkpoint(
                    Path(self.config.checkpoint_dir) / "best.pt",
                    epoch=epoch,
                    loss=val_loss,
                )
            self.save_training_checkpoint(
                Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt",
                epoch=epoch,
                loss=val_loss,
            )
            self.save_training_checkpoint(
                self.config.checkpoint_path,
                epoch=epoch,
                loss=val_loss,
            )
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, train_loss, val_loss)
            # Checkpoints have recorded the completed transition as
            # ``next_epoch``. The in-memory loop can now begin that epoch.
            self._resume_phase = "train"
        for callback in self.callbacks:
            callback.on_train_end(self)
