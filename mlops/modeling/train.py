from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    features_path = Path(cfg.paths.processed) / "features.csv"
    labels_path = Path(cfg.paths.processed) / "labels.csv"
    model_path = Path(cfg.paths.models) / "model.pkl"

    logger.info("Training some model...")
    logger.info(f"Features: {features_path} | Labels: {labels_path} | Model: {model_path}")

    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    main()
