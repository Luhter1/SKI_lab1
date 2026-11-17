import os
import shutil
from icrawler.builtin import BingImageCrawler

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

RUSSIAN_QUERIES = {
    "airplane": "самолет фото",
    "automobile": "легковой автомобиль фото",
    "bird": "птица фото",
    "cat": "кошка фото",
    "deer": "олень фото",
    "dog": "собака фото",
    "frog": "лягушка фото",
    "horse": "лошадь фото",
    "ship": "корабль фото",
    "truck": "грузовик фото",
}

IMAGES_PER_CLASS = 5
OUTPUT_DIR = "internet_test_images"

for class_name in CLASSES:
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": class_dir},
        downloader_threads=4,
    )

    crawler.crawl(
        keyword=RUSSIAN_QUERIES[class_name],
        max_num=IMAGES_PER_CLASS * 3,  # запас на плохие/повторные результаты
        min_size=(200, 200),
        file_idx_offset="auto",
    )

print("Готово. Проверьте папку internet_test_images/")