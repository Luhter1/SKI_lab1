import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from train import SimpleCNN

# Порядок классов должен совпадать с индексами CIFAR-10
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SRC = "internet_test_images"
OUT = "data/my_test"
NORM = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def downscale_all():
    """Уменьшаем каждую картинку до 32x32 и кладём в data/my_test/<класс>."""
    for cls in CLASSES:
        src_dir = os.path.join(SRC, cls)
        out_dir = os.path.join(OUT, cls)
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(os.listdir(src_dir))
        for name in names:
            path = os.path.join(src_dir, name)
            if not os.path.isfile(path):
                continue
            img = Image.open(path).convert("RGB")
            img = img.resize((32, 32), Image.LANCZOS)
            img.save(os.path.join(out_dir, os.path.splitext(name)[0] + ".jpg"), "JPEG")
    print(f"Даунгрейд выполнен: {SRC}/ -> {OUT}/ (32x32)")


def load_set():
    xs, ys, paths = [], [], []
    for cls_idx, cls in enumerate(CLASSES):
        dir_ = os.path.join(OUT, cls)
        for name in sorted(os.listdir(dir_)):
            p = os.path.join(dir_, name)
            img = Image.open(p).convert("RGB")
            # нормализация как при обучении
            arr = np.asarray(img, dtype=np.float32) / 255.0
            for c, (m, s) in enumerate(zip(*NORM)):
                arr[..., c] = (arr[..., c] - m) / s
            xs.append(torch.tensor(arr).permute(2, 0, 1))
            ys.append(cls_idx)
            paths.append(p)
    return torch.stack(xs), torch.tensor(ys), paths


def main():
    downscale_all()
    xs, ys, paths = load_set()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/cifar10_cnn.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(xs.to(device)).argmax(1).cpu().numpy()
    ys_np = ys.numpy()

    overall = (preds == ys_np).mean()
    print(f"\nОбразцов в наборе: {len(ys_np)}")
    print(f"Accuracy (общая): {overall:.4f} ({int((preds==ys_np).sum())}/{len(ys_np)})")
    print("\nAccuracy по классам:")
    for i, cls in enumerate(CLASSES):
        mask = ys_np == i
        acc = (preds[mask] == ys_np[mask]).mean() if mask.sum() else float("nan")
        print(f"  {cls:<12} n={mask.sum():3d}  acc={acc:.3f}")
    print("\nОтчёт (precision/recall/F1):")
    print(classification_report(ys_np, preds, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(ys_np, preds, labels=range(len(CLASSES)))
    os.makedirs("plots", exist_ok=True)
    _plot_grid(xs, ys_np, preds, paths, "plots/predictions_mytest.png")
    with open("metrics_my_test.json", "w") as f:
        json.dump({"overall_acc": float(overall), "n": int(len(ys_np)),
                   "per_class_acc": {cls: float((preds[ys_np == i] == ys_np[ys_np == i]).mean())
                                     for i, cls in enumerate(CLASSES)}},
                  f, indent=2, ensure_ascii=False)
    print("\nСохранено: plots/predictions_mytest.png, metrics_my_test.json")


def _plot_grid(xs, ys, preds, paths, out):
    """Сетка примеров: предсказание (зелёное/красное) и истинный класс."""
    idx = np.arange(len(ys))
    # покажем смесь верных и ошибочных примеров (до 12)
    wrong = idx[preds != ys]
    right = idx[preds == ys]
    sel = list(right[:max(0, 12 - min(3, len(wrong)))]) + list(wrong[:3])
    sel = (sel[:12])
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, i in zip(axes.flat, sel):
        # обратная нормализация для отображения
        img = xs[i].permute(1, 2, 0).numpy()
        for c, (m, s) in enumerate(zip(*NORM)):
            img[..., c] = img[..., c] * s + m
        img = np.clip(img, 0, 1)
        correct = preds[i] == ys[i]
        ax.imshow(img)
        ax.set_title(f"Предсказание: {CLASSES[preds[i]]}", fontsize=10,
                     color="green" if correct else "red")
        ax.set_xlabel(f"Истинный: {CLASSES[ys[i]]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Собственный тестовый набор (зелёный — верно, красный — ошибка)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()