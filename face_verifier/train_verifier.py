import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from face_verifier.dataset import FaceBinaryDataset
from face_verifier.model import FaceBinaryClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train binary face verifier")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--val_list", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./weights/verifier")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=1.0, help="BCE positive-class weight")
    parser.add_argument("--balanced_sampler", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(dataset: FaceBinaryDataset, batch_size: int, num_workers: int, balanced_sampler: bool, shuffle: bool) -> DataLoader:
    sampler = None
    if balanced_sampler:
        labels = [s.label for s in dataset.samples]
        counts = [max(1, labels.count(0)), max(1, labels.count(1))]
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    fpr = fp / (fp + tn + eps)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, Dict[str, float]]:
    model.train(train)
    total_loss = 0.0

    all_logits = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(logits, targets)
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"

    train_set = FaceBinaryDataset(
        root_dir=args.train_root,
        list_file=args.train_list,
        image_size=args.image_size,
        margin=args.margin,
        augment=True,
    )
    val_set = FaceBinaryDataset(
        root_dir=args.val_root,
        list_file=args.val_list,
        image_size=args.image_size,
        margin=args.margin,
        augment=False,
    )

    train_loader = build_loader(train_set, args.batch_size, args.num_workers, args.balanced_sampler, shuffle=True)
    val_loader = build_loader(val_set, args.batch_size, args.num_workers, False, shuffle=False)

    model = FaceBinaryClassifier(backbone=args.backbone).to(device)

    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, amp_enabled=amp_enabled, scaler=scaler
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, amp_enabled=amp_enabled, scaler=scaler
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f} val_fpr={val_metrics['fpr']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last.pth")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, save_dir / "best.pth")
            print(f"Saved best model @ epoch {epoch}, val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
