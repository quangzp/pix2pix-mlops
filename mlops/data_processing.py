from pathlib import Path
import shutil
from typing import Optional

from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import typer

app = typer.Typer()


def load_config(config_path: Path):
    """Load OmegaConf YAML config."""
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def process_image(input_path: Path, output_path: Path, size: Optional[int] = None):
    """Process a single image (resize, convert RGB, save)."""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")

            if size is not None:
                img = img.resize((size, size))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)

        return True

    except Exception as e:
        logger.error(f"Failed processing {input_path}: {e}")
        return False


@app.command()
def main(config: Path = typer.Option(..., help="Path to dataset config YAML")):
    """Run dataset processing pipeline."""
    cfg = load_config(config)

    raw_data = Path(cfg.dataset.raw_dir)
    processed_sketch = Path(cfg.dataset.processed_sketch_dir)
    processed_image = Path(cfg.dataset.processed_image_dir)
    size = cfg.dataset.resize

    # Clean processed dirs
    if cfg.dataset.clean_processed:
        logger.info("Cleaning processed directory...")
        shutil.rmtree(processed_sketch, ignore_errors=True)
        shutil.rmtree(processed_image, ignore_errors=True)

    processed_sketch.mkdir(parents=True, exist_ok=True)
    processed_image.mkdir(parents=True, exist_ok=True)

    sketch_dir = raw_data / "sketches"
    image_dir = raw_data / "images"

    sketch_paths = sorted(sketch_dir.glob("*"))
    image_paths = sorted(image_dir.glob("*"))

    logger.info(f"Found {len(sketch_paths)} sketches and {len(image_paths)} images.")

    success = 0

    for sketch_path, image_path in tqdm(zip(sketch_paths, image_paths), total=len(sketch_paths)):
        sketch_out = processed_sketch / sketch_path.name
        image_out = processed_image / image_path.name

        ok1 = process_image(sketch_path, sketch_out, size)
        ok2 = process_image(image_path, image_out, size)

        if ok1 and ok2:
            success += 1

    logger.success(f"Processing complete. {success}/{len(sketch_paths)} pairs processed.")


if __name__ == "__main__":
    app()
