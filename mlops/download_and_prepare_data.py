import glob
import os
from pathlib import Path
import shutil

import kagglehub


def prepare_data(n=200):
    print("Downloading/Checking dataset via KaggleHub...")
    edges2shoes_path = kagglehub.dataset_download("balraj98/edges2shoes-dataset")
    print(f"Dataset path: {edges2shoes_path}")

    train_path = os.path.join(edges2shoes_path, "train")
    valid_path = os.path.join(edges2shoes_path, "val")

    train_files = glob.glob(os.path.join(train_path, "*.jpg"))
    val_files = glob.glob(os.path.join(valid_path, "*.jpg"))

    total_files = train_files + val_files
    print(f"Total files found: {len(total_files)}")

    # Lấy n ảnh đầu tiên (hoặc random n ảnh)
    selected_files = train_files[:n]

    out_dir = Path("data/raw/original")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(selected_files):
        shutil.copy(img_path, out_dir / f"{i:04d}.jpg")
    print(f"Copied {len(selected_files)} original images to {out_dir}")


if __name__ == "__main__":
    prepare_data(n=200)
