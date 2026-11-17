import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from train import SimpleCNN, get_loaders

CLASSES = torchvision.datasets.CIFAR10("data", train=False).classes


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        ys.extend(y.tolist())
        preds.extend(model(x).argmax(1).tolist())
    return np.array(ys), np.array(preds)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок (test)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(rows, path):
    epochs = [r["epoch"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train loss")
    ax1.set_xlabel("Эпоха"); ax1.set_ylabel("Loss"); ax1.set_title("Функция потерь")
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(epochs, [r["train_acc"] for r in rows], marker="o", label="train acc")
    ax2.plot(epochs, [r["val_acc"] for r in rows], marker="s", label="val acc")
    ax2.set_xlabel("Эпоха"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy train vs val")
    ax2.grid(True, alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs("plots", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/cifar10_cnn.pt", map_location=device))

    _, val_loader, test_loader = get_loaders(batch_size=128)
    val_acc = sum(pred == y for y, pred in zip(*predict(model, val_loader))) / len(val_loader.dataset)

    y_test, p_test = predict(model, test_loader)
    test_acc = (y_test == p_test).mean()

    report = classification_report(y_test, p_test, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion(cm, "plots/confusion_matrix.png")
    plot_curves(_load_history(), "plots/training_curves.png")

    metrics = {"val_acc": float(val_acc), "test_acc": float(test_acc),
               "test_samples": int(len(y_test)),
               "classification_report": report}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Accuracy на валидационной выборке: {val_acc:.4f}")
    print(f"Accuracy на тестовой выборке:      {test_acc:.4f}  (n={len(y_test)})")
    print("\nОтчёт по классам (test):\n")
    print(classification_report(y_test, p_test, target_names=CLASSES))
    print("Графики сохранены в plots/, метрики в metrics.json")


def _load_history():
    with open("training_history.csv") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()