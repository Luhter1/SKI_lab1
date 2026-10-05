import numpy as np

def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Прямая 2D-свёртка одного канала с отражением границ"""
    size = kernel.shape[0]
    pad = size // 2
    h, w = channel.shape

    # добавляем отражение для матрицы на границе
    padded = np.pad(channel, pad, mode="reflect")
    result = np.zeros((h, w), dtype=np.float64)

    # Двойной цикл по элементам ядра — «нативный» подход без встроенных
    # функций свёртки
    for i in range(size):
        for j in range(size):
            k = kernel[i, j]
            if k == 0.0:
                continue
            result += k * padded[i:i + h, j:j + w]

    return result


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Возвращает нормированное 2D-ядро Гаусса размера size x size
    G(x,y) = e^(-(x^2 + y^2) / (2 * sigma^2))
    """
    if size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    if sigma <= 0:
        raise ValueError("sigma должна быть > 0")

    # Строим координатную ось от -k до +k (size=5, k=2, ax=[-2, -1, 0, 1, 2])
    ax = np.arange(-(size // 2), size // 2 + 1)
    # xx =            yy =
    # [-2 -1 0 1 2]   [-2 -2 -2 -2 -2]
    # [-2 -1 0 1 2]   [-1 -1 -1 -1 -1]
    # [-2 -1 0 1 2]   [ 0  0  0  0  0]
    # [-2 -1 0 1 2]   [ 1  1  1  1  1]
    # [-2 -1 0 1 2]   [ 2  2  2  2  2]
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_native(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Размытие изображения фильтром Гаусса (нативная реализация)"""
    kernel = gaussian_kernel(size, sigma)
    img = image.astype(np.float64)

    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            out[:, :, c] = _convolve_channel(img[:, :, c], kernel)
    else:
        raise ValueError("Ожидалось 3D изображение")

    return np.clip(out, 0, 255).astype(np.uint8)