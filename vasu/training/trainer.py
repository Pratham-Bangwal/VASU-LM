from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer
from vasu.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from vasu.training.sample import generate_sample

class Trainer:

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        config,
        device,
        callbacks=None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.callbacks = callbacks or []

        self.loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
        )

        self.optimizer = build_optimizer(
            self.model,
            config,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        self.best_val_loss = float("inf")
        


        checkpoint = Path("checkpoints/vasu.pt")

        self.start_epoch = 0

        if checkpoint.exists():

            ckpt = torch.load(
                checkpoint,
                map_location=device,
            )

            self.model.load_state_dict(
                ckpt["model"]
            )

            self.optimizer.load_state_dict(
                ckpt["optimizer"]
            )

            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(
                    ckpt["scheduler"]
                )

            self.start_epoch = ckpt["epoch"] + 1

            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

            print(
                f"\n✅ Resuming from epoch {self.start_epoch}\n"
            )
    

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        progress = tqdm(
            self.loader,
            desc="Training",
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(progress):

            x = x.to(self.device)
            y = y.to(self.device)


            with torch.cuda.amp.autocast():

                logits = self.model(x)

                loss = language_model_loss(
                    logits,
                    y,
                )

                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (
                (step + 1) % self.config.gradient_accumulation_steps == 0
                or (step + 1) == len(self.loader)
            ):

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            total_loss += (
                loss.item()
                * self.config.gradient_accumulation_steps
        )

            progress.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}"
            })
            

        return total_loss / len(self.loader)

    @torch.no_grad()
    def validate_epoch(self):

        self.model.eval()

        total_loss = 0.0

        for x, y in self.val_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            with torch.cuda.amp.autocast():

                logits = self.model(x)

                loss = language_model_loss(
                    logits,
                    y,
                )

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self):


        Path("checkpoints").mkdir(
            exist_ok=True
        )

        for callback in self.callbacks:
            callback.on_train_begin(self)

        print(f"start_epoch = {self.start_epoch}")
        print(f"epochs = {self.config.epochs}")

        for epoch in range(
            self.start_epoch,
            self.config.epochs,
        ):

            for callback in self.callbacks:
                callback.on_epoch_begin(
                    self,
                    epoch,
                )

            train_loss = self.train_epoch()

            val_loss = self.validate_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(
                    self,
                    epoch,
                    train_loss,
                    val_loss,
                )


            self.scheduler.step()

            print(
                f"Epoch {epoch+1} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_val_loss": self.best_val_loss,
                    },
                    "checkpoints/best.pt",
                )

                print("⭐ New best model saved!")

            # after each epoch
            
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_loss,
                f"checkpoints/epoch_{epoch+1}.pt",
            )
            generate_sample(
                self.model,
                self.tokenizer,
                self.device,
            )

        for callback in self.callbacks:
            callback.on_train_end(self)