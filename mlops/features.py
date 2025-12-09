from pathlib import Path
from PIL import Image
from loguru import logger
from tqdm import tqdm
import typer

from mlops.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def load_image(path: Path, size = (256,256)):
    """Load and preprocess an image from the given path."""
    image = Image.open(path)
    image = image.resize(size)
    image = image.convert("RGB")
    return image

@app.command()
def main(
    sketch_path: Path = PROCESSED_DATA_DIR / "sketch",
    photo_path: Path = PROCESSED_DATA_DIR / "photo",
    output_features_path: Path = INTERIM_DATA_DIR / "pairs.json"
):

    logger.info("Generating features from dataset...")
  
    pairs = []
    sketch_files = sorted(sketch_path.glob("*.jpg"))
    logger.info(f"Found {len(sketch_files)} sketch files.")

    for sketch_file in tqdm(sketch_files):
        base = sketch_file.stem
        real_image = photo_path / f"{base}.jpg"
        if not real_image.exists():
            logger.warning(f"Real image not found for sketch: {sketch_file.name}")
            continue
        try:
            load_image(sketch_file)
            load_image(real_image)

            pairs.append({
                "sketch_path": str(sketch_file),
                "real_path": str(real_image),
            })
        except Exception as e:
            logger.error(f"Error processing {sketch_file.name}: {e}")
        
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_features_path, "w") as f:
        import json
        json.dump(pairs, f, indent=2)
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
