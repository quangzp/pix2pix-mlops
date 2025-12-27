from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    input_path = Path(cfg.paths.processed) / "dataset.csv"
    output_path = Path(cfg.paths.figures) / "plot.png"

    logger.info(f"Input: {input_path} | Output: {output_path}")

    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    main()
