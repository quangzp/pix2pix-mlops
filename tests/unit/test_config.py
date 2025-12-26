from pathlib import Path

from hydra import compose, initialize


def test_hydra_config_load():
    with initialize(version_base=None, config_path="../../mlops/config"):
        cfg = compose(config_name="config")

        print(f"Loaded paths: {cfg.paths}")

        assert isinstance(cfg.paths.raw, str)
        assert isinstance(cfg.paths.processed, str)
        assert isinstance(cfg.paths.models, str)

        raw_path = Path(cfg.paths.raw)
        assert raw_path.parent.exists() or raw_path.exists()

        assert cfg.dataset is not None
        assert cfg.training is not None
        assert cfg.model is not None


if __name__ == "__main__":
    test_hydra_config_load()
