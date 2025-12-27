from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="mlops/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Expose shortcuts
    raw_data_dir = Path(cfg.paths.raw)
    processed_data_dir = Path(cfg.paths.processed)
    interim_data_dir = Path(cfg.paths.interim)
    models_dir = Path(cfg.paths.models)

    dataset_cfg = cfg.dataset
    training_cfg = cfg.training
    model_cfg = cfg.model

    print("Raw data dir:", raw_data_dir)
    print("Processed data dir:", processed_data_dir)
    print("Models dir:", models_dir)
    print("Dataset config:", dataset_cfg)
    print("Training config:", training_cfg)
    print("Model config:", model_cfg)
    print("Interim data dir:", interim_data_dir)


if __name__ == "__main__":
    main()
