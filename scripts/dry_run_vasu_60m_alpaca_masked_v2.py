"""Run the masked-v2 preflight without modifying experiment checkpoints."""

import os
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.checkpoint import save_checkpoint
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer


BASE_CHECKPOINT = Path(
    "checkpoints/vasu_60m/milestones/fineweb_step_150000.pt"
)
DATA_FILE = Path("data/processed/instruct/alpaca_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/alpaca_masked_v2_mask.bin")
TEMP_DIR = Path("checkpoints/vasu_60m/alpaca_masked_v2_dry_run")
TEMP_CHECKPOINT = TEMP_DIR / "dry_run.pt"
SEED = 42


def gradients_are_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VASU-60M dry run")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
        config = get_vasu_60m_config()
        dataset = PackedInstructionDataset(
            str(DATA_FILE), str(MASK_FILE), config.max_seq_len
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        inputs, targets, mask = next(iter(loader))
        supervised_tokens = int(mask.sum().item())
        if supervised_tokens == 0:
            raise RuntimeError("dry-run batch has zero supervised tokens")

        tokenizer = VASUTokenizer()
        tokenizer.load("assets/tokenizer.json")
        masked_ids = targets[0][mask[0].bool()].tolist()
        print(f"Token shape: {tuple(inputs.shape)}")
        print(f"Mask shape: {tuple(mask.shape)}")
        print(f"Supervised tokens: {supervised_tokens}")
        print(f"Assistant-token ratio: {mask.mean().item():.6f}")
        print(f"Decoded masked region: {tokenizer.decode(masked_ids)!r}")

        train_config = TrainConfig(
            epochs=1,
            batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,
            weight_decay=0.01,
            grad_clip=1.0,
            use_amp=True,
        )
        model = VASUModel(config).to(device)
        base = torch.load(BASE_CHECKPOINT, map_location="cpu")
        model.load_state_dict(base["model"])
        del base
        optimizer = build_optimizer(model, train_config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(inputs)
            loss = language_model_loss(logits, targets, mask)
        if not torch.isfinite(loss):
            raise RuntimeError("dry-run loss is not finite")
        scaler.scale(loss / 16).backward()
        scaler.unscale_(optimizer)
        if not gradients_are_finite(model):
            raise RuntimeError("dry-run gradients are not finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        temporary = TEMP_CHECKPOINT.with_suffix(".pt.tmp")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=0,
            loss=float(loss.item()),
            path=str(temporary),
            global_step=150001,
        )
        os.replace(temporary, TEMP_CHECKPOINT)
        restored = torch.load(TEMP_CHECKPOINT, map_location="cpu")
        required = {
            "epoch", "global_step", "model", "optimizer", "scheduler", "loss"
        }
        if not required.issubset(restored):
            raise RuntimeError("dry-run checkpoint schema validation failed")
        if int(restored["global_step"]) != 150001:
            raise RuntimeError("dry-run resume global_step validation failed")

        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Loss: {loss.item():.6f}")
        print("Finite gradients: True")
        print("Checkpoint save/load: True")
        print("Resume global_step: 150001")
        print(f"Peak CUDA memory: {peak_memory:.2f} MiB")
    finally:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f"Temporary dry-run checkpoints deleted: {TEMP_DIR}")


if __name__ == "__main__":
    main()
