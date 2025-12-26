from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    features_path = Path(cfg.paths.processed) / "test_features.csv"
    model_path = Path(cfg.paths.models) / "model.pkl"
    predictions_path = Path(cfg.paths.processed) / "test_predictions.csv"

    logger.info(f"Performing inference for model: {model_path}")
    logger.info(f"Features: {features_path} | Predictions: {predictions_path}")

    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")


if __name__ == "__main__":
    main()
