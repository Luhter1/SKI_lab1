import time
import numpy as np
import matplotlib.pyplot as plt

from src import (
    gaussian_blur_native,
    gaussian_blur_opencv,
)


def timeit(fn, *args, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def run():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    sizes = [3, 5, 7, 9, 15, 21]
    sigma_ratio = 0.3  # sigma = ratio * size

    results = {"native": [], "separable": [], "opencv": []}

    print(f"{'size':>5} | {'native, ms':>10} | {'separ, ms':>10} | {'opencv, ms':>11} | speedup(native/opencv)")
    print("-" * 70)
    for s in sizes:
        sigma = sigma_ratio * s
        t_nat = timeit(gaussian_blur_native, img, s, sigma) * 1000
        t_cv = timeit(gaussian_blur_opencv, img, s, sigma) * 1000
        results["native"].append(t_nat)
        results["opencv"].append(t_cv)
        print(f"{s:>5} | {t_nat:>10.2f} | {t_cv:>11.2f} | x{t_nat / t_cv:>8.1f}")

    # График
    plt.figure(figsize=(9, 5))
    plt.plot(sizes, results["native"], "o-", label="Native (naive)")
    plt.plot(sizes, results["opencv"], "^-", label="OpenCV")
    plt.yscale("log")
    plt.xlabel("Размер ядра, px")
    plt.ylabel("Время, мс (log)")
    plt.title("Время выполнения фильтра Гаусса (512×512×3)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=150)
    print("\nГрафик сохранён: benchmark.png")


if __name__ == "__main__":
    run()