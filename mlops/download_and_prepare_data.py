import glob
import os
from pathlib import Path

import kagglehub
from PIL import Image
from sklearn.model_selection import train_test_split


def split_and_save(img_list, out_sketch, out_real, n=200):
    out_sketch.mkdir(parents=True, exist_ok=True)
    out_real.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(img_list[:n]):
        img = Image.open(img_path)
        w, h = img.size
        sketch = img.crop((0, 0, w // 2, h))
        real = img.crop((w // 2, 0, w, h))
        sketch.save(out_sketch / f"{i:04d}.jpg")
        real.save(out_real / f"{i:04d}.jpg")


def prepare_data(n=200):
    print("Downloading/Checking dataset via KaggleHub...")
    # Lưu ý: Lệnh này sẽ tải về thư mục cache của user máy tính (~/.cache/kagglehub/...)
    edges2shoes_path = kagglehub.dataset_download("balraj98/edges2shoes-dataset")
    print(f"Dataset path: {edges2shoes_path}")

    train_path = os.path.join(edges2shoes_path, "train")
    valid_path = os.path.join(edges2shoes_path, "val")

    train_files = glob.glob(os.path.join(train_path, "*.jpg"))
    val_files = glob.glob(os.path.join(valid_path, "*.jpg"))

    total_files = train_files + val_files
    print(f"Total files found: {len(total_files)}")

    # Split Data
    train_list, temp_list = train_test_split(
        total_files, test_size=0.2, random_state=42, shuffle=True
    )
    val_list, test_list = train_test_split(temp_list, test_size=0.5, random_state=42, shuffle=True)

    print(f"Train: {len(train_list)} | Val: {len(val_list)} | Test: {len(test_list)}")

    # Save only n samples cho demo
    split_and_save(train_list, Path("data/raw/train/sketch"), Path("data/raw/train/photo"), n)
    split_and_save(val_list, Path("data/raw/val/sketch"), Path("data/raw/val/photo"), n // 5)
    split_and_save(test_list, Path("data/raw/test/sketch"), Path("data/raw/test/photo"), n // 5)
    print("Demo dataset prepared at data/raw/[train|val|test]/[sketch|photo]")


if __name__ == "__main__":
    prepare_data(n=200)
