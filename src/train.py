from __future__ import annotations

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES, GestureSmokeDataset, prepare_smoke_dataset
from model import GatedConvSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DVS gesture SNN+gate smoke experiment")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-lambda", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--models", default="baseline,gated")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_matrix(targets: list[int], predictions: list[int], classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    f1_scores = []
    for class_id in range(matrix.shape[0]):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1_scores.append(2 * tp / denominator if denominator else 0.0)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_scores))}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []
    gate_values: list[float] = []
    active_gate_values: list[float] = []
    spike_rates: list[float] = []
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, aux = model(x)
        losses.append(float(criterion(logits, y).item()))
        prediction = logits.argmax(dim=1)
        targets.extend(y.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        gates = aux["gates"]
        gate_values.append(float(gates.mean().item()))
        active_gate_values.append(float((gates >= 0.5).float().mean().item()))
        spike_rates.append(float(aux["spike_rates"].mean().item()))

    matrix = confusion_matrix(targets, predictions, len(CLASSES))
    result = metrics_from_confusion(matrix)
    result.update(
        {
            "loss": float(np.mean(losses)),
            "gate_mean": float(np.mean(gate_values)),
            "gate_active_ratio_0.5": float(np.mean(active_gate_values)),
            "mean_spike_rate": float(np.mean(spike_rates)),
            "confusion_matrix": matrix.tolist(),
            "targets": targets,
            "predictions": predictions,
        }
    )
    return result


def train_one(
    name: str,
    gate_enabled: bool,
    loaders: dict[str, DataLoader],
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    seed_everything(args.seed)
    model = GatedConvSNN(num_classes=len(CLASSES), gate_enabled=gate_enabled).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = 0
    best_val_accuracy = -1.0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(x)
            classification_loss = criterion(logits, y)
            gate_cost = aux["gates"].mean() if gate_enabled else logits.new_zeros(())
            loss = classification_loss + args.gate_lambda * gate_cost
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_metrics = evaluate(model, loaders["train_eval"], device)
        val_metrics = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "optimization_loss": float(np.mean(epoch_losses)),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_gate_mean": val_metrics["gate_mean"],
            "val_spike_rate": val_metrics["mean_spike_rate"],
        }
        history.append(row)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[{name}] epoch={epoch:03d} train_acc={train_metrics['accuracy']:.3f} "
                f"val_acc={val_metrics['accuracy']:.3f} gate={val_metrics['gate_mean']:.3f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, loaders["train_eval"], device)
    val_metrics = evaluate(model, loaders["val"], device)
    test_metrics = evaluate(model, loaders["test"], device)
    elapsed = time.perf_counter() - started

    model_path = output_dir / f"{name}_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": CLASSES,
            "gate_enabled": gate_enabled,
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        model_path,
    )

    history_path = output_dir / f"{name}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    return {
        "name": name,
        "gate_enabled": gate_enabled,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "runtime_seconds": elapsed,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "checkpoint": model_path.name,
        "history": history_path.name,
    }


def save_plots(results: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.4))
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        matrix = np.asarray(result["test"]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        axis.set_title(f"{result['name']} test confusion")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
        axis.set_yticks(range(len(CLASSES)), CLASSES)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrices.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for result in results:
        history = np.genfromtxt(
            output_dir / result["history"], delimiter=",", names=True, encoding="utf-8-sig"
        )
        axis.plot(history["epoch"], history["train_accuracy"], label=f"{result['name']} train")
        axis.plot(history["epoch"], history["val_accuracy"], linestyle="--", label=f"{result['name']} val")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_dir = root / "data" / "raw" / "zenodo_3228846_smoke"
    processed_path = root / "data" / "processed" / "smoke_v1.npz"
    output_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = prepare_smoke_dataset(raw_dir, processed_path)
    datasets = {
        split: GestureSmokeDataset(processed_path, split, augment=(split == "train"))
        for split in ("train", "val", "test")
    }
    train_eval_dataset = GestureSmokeDataset(processed_path, "train", augment=False)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed), num_workers=0,
        ),
        "train_eval": DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    model_specs = {
        "baseline": ("baseline", False),
        "gated": ("gated", True),
    }
    results = []
    for requested_name in requested:
        if requested_name not in model_specs:
            raise ValueError(f"Unknown model {requested_name}")
        name, gate_enabled = model_specs[requested_name]
        results.append(train_one(name, gate_enabled, loaders, device, args, output_dir))

    save_plots(results, output_dir)
    summary = {
        "experiment": "DVS hand-gesture gated-SNN smoke",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope_warning": "Small four-session smoke only; not a full benchmark result.",
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "gate_lambda": args.gate_lambda,
        "dataset": dataset_summary,
        "splits": {name: len(dataset) for name, dataset in datasets.items()},
        "models": results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
