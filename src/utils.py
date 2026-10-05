import numpy as np
import cv2


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str, image: np.ndarray) -> None:
    if image.ndim == 3: # в цветном изображении 3 канала
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    # среднеквадратичная ошибка
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    # пиковое отношение сигнал/шум
    # > 50 dB Изображения практически идентичны
    # 40—50 dB Очень хорошее качество, отличие незаметно глазом
    # 30—40 dB Хорошее качество, лёгкие артефакты
    # 20—30 dB Заметные искажения
    # < 20 dB Сильные искажения, сравнение лишено смысла
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / m)