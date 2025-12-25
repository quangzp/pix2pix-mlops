from pathlib import Path

from omegaconf import OmegaConf

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"

cfg = OmegaConf.load(CONFIG_PATH)

# Expose shortcuts
RAW_DATA_DIR = Path(cfg.paths.raw)
PROCESSED_DATA_DIR = Path(cfg.paths.processed)
INTERIM_DATA_DIR = Path(cfg.paths.interim)
MODELS_DIR = Path(cfg.paths.models)

DATASET_CFG = cfg.dataset
TRAINING_CFG = cfg.training
MODEL_CFG = cfg.model
