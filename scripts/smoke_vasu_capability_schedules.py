"""Run bounded, non-optimizing VASU-60M schedule smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vasu.config import get_vasu_60m_config
from vasu.data.scheduled_mixture import ScheduledPretrainingDataset
from vasu.model.model import VASUModel
from vasu.training.losses import language_model_loss


PARENT_CHECKPOINT = Path(
    "checkpoints/vasu_60m/milestones/fineweb_step_200000.pt"
)
RELEASE_ROOT = Path("data/processed/capability/mixtures")
CANDIDATES = (
    "capability_cpt_a_factual_20m_v1",
    "capability_cpt_b_balanced_20m_v1",
    "capability_cpt_c_control_20m_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Required opt-in for bounded forward/backward smoke checks.",
    )
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    return parser.parse_args()


def representative_indices(
    dataset: ScheduledPretrainingDataset,
    *,
    target_count: int = 32,
) -> list[int]:
    wanted = {source.identifier for source in dataset.sources}
    selected = {0, len(dataset) // 2, len(dataset) - 1}
    for index in range(len(dataset)):
        source_id, _ = dataset.source_identity(index)
        if source_id in wanted:
            selected.add(index)
            wanted.remove(source_id)
        if not wanted:
            break
    for index in range(len(dataset)):
        selected.add(index)
        if len(selected) >= target_count:
            break
    return sorted(selected)


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Pass --run for the bounded smoke; training is never started.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    checkpoint = torch.load(
        PARENT_CHECKPOINT,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    results: dict[str, object] = {}
    for candidate in CANDIDATES:
        candidate = candidate.removesuffix("_v1") + f"_{args.version}"
        manifest_path = RELEASE_ROOT / candidate / "resolved_manifest.json"
        dataset = ScheduledPretrainingDataset.from_resolved_manifest(manifest_path)
        indices = representative_indices(dataset)
        identities = [dataset.source_identity(index) for index in indices]
        loader = DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        model = VASUModel(get_vasu_60m_config()).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.1,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for inputs, targets, mask in loader:
            with torch.amp.autocast(
                device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs.to(device))
                loss = language_model_loss(
                    logits,
                    targets.to(device),
                    mask.to(device),
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite smoke loss for {candidate}")
            scaler.scale(loss / 16).backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite gradients for {candidate}")
            losses.append(float(loss.detach().cpu()))
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        results[candidate] = {
            "records": len(dataset),
            "representative_indices": indices,
            "source_identities": identities,
            "losses": losses,
            "finite": True,
            "optimizer_steps": 1,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["environment"] = {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    results["optimizer_updates"] = len(CANDIDATES)
    results["training_started"] = False
    results["peak_cuda_mib"] = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else None
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
