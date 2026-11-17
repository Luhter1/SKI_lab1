"""Обучение свёрточной нейронной сети для классификации CIFAR-10.

Запуск: .venv/bin/python train.py [--epochs N] [--batch-size N]
Результаты: models/cifar10_cnn.pt, training_history.csv
"""
import argparse
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# --- Архитектура сети -------------------------------------------------------
class SimpleCNN(nn.Module):
    """Небольшая свёрточная сеть для изображений 32x32x3."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --- Данные -----------------------------------------------------------------
def get_loaders(batch_size, val_ratio=0.1, seed=42):
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), norm])
    eval_tf = T.Compose([T.ToTensor(), norm])

    train_full = torchvision.datasets.CIFAR10("data", train=True, download=True,
                                              transform=train_tf)
    eval_full = torchvision.datasets.CIFAR10("data", train=False, download=True,
                                             transform=eval_tf)

    n_val = int(len(train_full) * val_ratio)
    train_set, val_set = random_split(
        train_full, [len(train_full) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(eval_full, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# --- Цикл обучения ----------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device} | эпох: {args.epochs} | batch: {args.batch_size}")

    train_loader, val_loader, test_loader = get_loaders(args.batch_size)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    rows = []
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)
        rows.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                     "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "models/cifar10_cnn.pt")
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f} "
              f"| {time.time()-t0:.1f}s")

    with open("training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово за {time.time()-start:.0f}s. Лучший val_acc: {best_val:.4f}")
    print("Модель сохранена в models/cifar10_cnn.pt")


if __name__ == "__main__":
    main()