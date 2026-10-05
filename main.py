import argparse
import os

from src import (
    gaussian_blur_native,
    gaussian_blur_opencv,
    load_image,
    save_image,
    mse,
    psnr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian Blur Demo")
    parser.add_argument("--image", required=True, help="путь к изображению")
    parser.add_argument("--size", type=int, default=7, help="размер ядра")
    parser.add_argument("--sigma", type=float, default=2.0, help="sigma гауссианы")
    parser.add_argument("--out", default="out", help="папка для результатов")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    img = load_image(args.image)
    print(f"Изображение: {img.shape}, dtype={img.dtype}")

    native = gaussian_blur_native(img, args.size, args.sigma)
    opencv = gaussian_blur_opencv(img, args.size, args.sigma)

    save_image(os.path.join(args.out, "native.png"), native)
    save_image(os.path.join(args.out, "opencv.png"), opencv)

    print(f"PSNR(native vs opencv)   = {psnr(native, opencv):.2f} dB")
    print(f"MSE (native vs opencv)   = {mse(native, opencv):.4f}")


if __name__ == "__main__":
    main()