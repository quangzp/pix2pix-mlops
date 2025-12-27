import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="mlops/config", config_name="config")
def test_config(cfg: DictConfig):
    print("Main Config")
    print(f"Project: {cfg.project_name}")
    print(f"Version: {cfg.version}")
    print(f"Seed: {cfg.seed}")
    print(f"Device: {cfg.device}")

    print("Paths Config")
    try:
        print(f"Root: {cfg.paths.root}")
        print(f"Raw data: {cfg.paths.raw}")
        print(f"Processed: {cfg.paths.processed}")
        print(f"Models: {cfg.paths.models}")
    except Exception as e:
        print(f"Paths config error: {e}")

    print("Dataset Config")
    try:
        print(f"Name: {cfg.dataset.name}")
        print(f"Image Size: {cfg.dataset.image_size}")
        print(f"Batch size: {cfg.dataset.batch_size}")
        print("Done config dataset")
    except Exception as e:
        print(f"Dataset config error: {e}")

    print("Model Config")
    try:
        print(f"Name: {cfg.model.name}")
        print(f"Generator ngf: {cfg.model.generator.ngf}")
        print(f"Discriminator ngf: {cfg.model.discriminator.ngf}")
        print("Done Model Config")
    except Exception as e:
        print(f"Model config error: {e}")

    print("Training Config")
    try:
        print(f"Epochs: {cfg.training.num_epochs}")
        print(f"Learning rate: {cfg.training.optimizer.lr}")
        print(f"Batch size: {cfg.training.batch_size}")
        print("Done Training Config")
    except Exception as e:
        print(f"Training config error: {e}")

    print("Params Config")
    try:
        metrics_count = len(cfg.params.metrics)
        print(f"Metrics: {metrics_count} defined")
        print("Done Params Config")
    except Exception as e:
        print(f"Params config error: {e}")

    loaded = 0
    total = 5

    configs = [
        ("paths", lambda: cfg.paths.root),
        ("dataset", lambda: cfg.dataset.name),
        ("model", lambda: cfg.model.name),
        ("training", lambda: cfg.training.num_epochs),
        ("params", lambda: cfg.params.metrics),
    ]

    for name, check_func in configs:
        try:
            check_func()
            loaded += 1
        except Exception as e:
            print(f"Error loading {name} config: {e}")

    print(f"Loaded {loaded}/{total} config sections successfully.")

    if loaded == total:
        print("All config sections loaded successfully.")
    else:
        print(f"Some config sections failed to load. {total - loaded} configs missing")


if __name__ == "__main__":
    test_config()
