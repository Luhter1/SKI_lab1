import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from train import SimpleCNN

CLASSES = torchvision.datasets.CIFAR10("data", train=False).classes
N_SAMPLES = 12


def main():
    os.makedirs("plots", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/cifar10_cnn.pt", map_location=device))
    model.eval()

    # Сырые изображения (для отображения) и те же с нормализацией (для предсказания)
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    raw = torchvision.datasets.CIFAR10("data", train=False, download=True)
    nrm = torchvision.datasets.CIFAR10("data", train=False, transform=T.Compose([T.ToTensor(), norm]))

    # Возьмём N_SAMPLES первых + отберём среди них и верные, и ошибочные
    idxs = list(range(N_SAMPLES))
    wrong = []
    for i in range(N_SAMPLES, len(raw)):
        if len(idxs) >= N_SAMPLES and len(wrong) >= 2:
            break
        img = nrm[i][0].unsqueeze(0)
        pred = model(img).argmax(1).item()
        if pred != raw[i][1] and len(wrong) < 2:
            idxs.append(i)
            wrong.append(i)
    idxs = idxs[:N_SAMPLES]

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, i in zip(axes.flat, idxs):
        x = nrm[i][0].unsqueeze(0)
        pred = model(x).argmax(1).item()
        true = raw[i][1]
        correct = pred == true

        img = raw[i][0]
        ax.imshow(img)
        ax.set_title(f"Предсказание: {CLASSES[pred]}", fontsize=10,
                     color="green" if correct else "red")
        ax.set_xlabel(f"Истинный класс: {CLASSES[true]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Примеры классификации (зелёный — верно, красный — ошибка)", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/predictions.png", dpi=150)
    plt.close(fig)
    print("Сохранено: plots/predictions.png")


if __name__ == "__main__":
    main()