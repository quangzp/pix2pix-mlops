from pathlib import Path
from typing import Optional

import hydra
from loguru import logger
import mlflow
from omegaconf import DictConfig
import torch
import wandb
from omegaconf import OmegaConf 

from mlops.src.components.discriminator import define_D
from mlops.src.components.generator import define_G
from mlops.src.components.losses import GANLoss, VGGLoss
from mlops.src.components.replay_pool import ReplayPool
from mlops.src.models.pix2pixhd_module import Pix2PixHD, Pix2PixHDDataset


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---- Dataset paths ----
    dataset_path: Path = Path(cfg.paths.raw)
    feature_folder = cfg.dataset.processed_sketch_dir
    label_folder = cfg.dataset.raw_dir
    # ---- Model configuration ----
    checkpoint_dir: Path = Path(cfg.checkpoints.dir)
    num_epochs: int = cfg.training.num_epochs
    batch_size: int = cfg.training.batch_size
    # ---- Generator config ----
    ngf = cfg.model.generator_channels
    n_downsample_global: int = cfg.generator.n_downsample_global
    n_blocks_global: int = cfg.generator.n_blocks_global
    n_local_enhancers: int = cfg.generator.n_local_enhancers
    n_blocks_local: int = cfg.generator.n_blocks_local
    # ---- Discriminator config ----
    ndf = cfg.model.discriminator_channels
    n_layers_D: int = cfg.discriminator.n_layers_D
    num_D: int = cfg.discriminator.num_D
    # ---- Training config ----
    learning_rate: float = cfg.training.learning_rate
    lambda_feat: float = cfg.training.lambda_feat
    replay_pool_size: int = cfg.training.replay_pool_size
    # ema_decay: float = cfg.training.ema_decay
    # ---- Data config ----
    img_size: int = cfg.data.img_size
    num_workers: int = cfg.data.num_workers
    # ---- Training control ----
    # test_interval: int = cfg.training.test_interval
    # save_interval: int = cfg.training.save_interval
    resume_from: Optional[Path] = cfg.training.resume_from

    """
    Train Pix2PixHD model for image-to-image translation.

    This training script initializes the generator and discriminator networks,
    sets up loss functions, and trains the model using the Pix2PixHD training pipeline.
    """

    # Setup logging
    logger.info("=" * 80)
    logger.info("Starting Pix2PixHD Training")
    logger.info("=" * 80)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # ---- Create dataset and dataloaders ----
        logger.info(f"Loading dataset from {dataset_path}")
        train_dataset = Pix2PixHDDataset(
            images_dir=str(dataset_path),
            feature_fold=feature_folder,
            label_fold=label_folder,
            img_size=img_size,
        )
        logger.info(f"Dataset size: {len(train_dataset)}")

        # Create train/test split
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        logger.info(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

        # ---- Create generator networks ----
        logger.info("Initializing generator networks...")
        generator = define_G(
            input_nc=3,
            output_nc=3,
            ngf=ngf,
            netG="global",
            norm="instance",
            n_downsample_global=n_downsample_global,
            n_blocks_global=n_blocks_global,
            n_local_enhancers=n_local_enhancers,
            n_blocks_local=n_blocks_local,
            gpu_ids=[],
        ).to(device)
        logger.success("Generator initialized")

        # ---- Create discriminator network ----
        logger.info("Initializing discriminator network...")
        discriminator = define_D(
            input_nc=6,  # 3 (input) + 3 (output)
            ndf=ndf,
            n_layers_D=n_layers_D,
            norm="instance",
            use_sigmoid=False,
            num_D=num_D,
            getIntermFeat=True,
            gpu_ids=[],
            num_outputs=1,
        ).to(device)
        logger.success("Discriminator initialized")

        # ---- Create loss functions ----
        logger.info("Initializing loss functions...")
        criterion_gan = GANLoss(use_lsgan=True).to(device)
        criterion_feat = torch.nn.L1Loss().to(device)
        criterion_vgg = VGGLoss().to(device)
        logger.success("Loss functions initialized")

        # ---- Create replay buffer ----
        logger.info(f"Initializing replay pool with size {replay_pool_size}...")
        replay_pool = ReplayPool(replay_pool_size)
        logger.success("Replay pool initialized")

        # ---- Initialize Pix2PixHD model ----
        logger.info("Initializing Pix2PixHD model...")
        model = Pix2PixHD(
            generator=generator,
            discriminator=discriminator,
            criterion_gan=criterion_gan,
            criterion_feat=criterion_feat,
            criterion_vgg=criterion_vgg,
            replay_pool=replay_pool,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            lambda_feat=lambda_feat,
        )
        logger.success("Pix2PixHD model initialized")

        # ---- Create optimizers ----
        logger.info(f"Creating optimizers with learning rate {learning_rate}...")
        g_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate)
        d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate)
        logger.success("Optimizers created")

        # ---- Load checkpoint if resuming ----
        start_epoch = 0
        if resume_from is not None:
            logger.info(f"Resuming training from checkpoint: {resume_from}")
            model.load_checkpoint(str(resume_from))
            logger.success("Checkpoint loaded")

        # ---- Training loop ----
        # 1. Setup MLflow
        mlflow.set_experiment(cfg.experiment.name)
        
        # 2. Setup WandB 
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        
        # Khởi tạo run
        run = wandb.init(
            entity=cfg.logger.wandb.entity,
            project=cfg.logger.wandb.project,
            group=cfg.logger.wandb.group,
            name=cfg.logger.wandb.name,
            config=wandb_config,
            job_type="training"
        )

        run_url = run.get_url()
        
        logger.info("=" * 80)
        logger.success(f"🚀 WANDB DASHBOARD IS LIVE AT: {run_url}")
        logger.info("=" * 80)

        with mlflow.start_run():
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("ngf", ngf)
            mlflow.log_param("ndf", ndf)
            logger.info("=" * 80)
            logger.info("Starting training loop")
            logger.info("=" * 80)

            for epoch in range(start_epoch, num_epochs):
                logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

                try:
                    model.train_epoch(
                        train_loader=train_loader,
                        test_loader=test_loader,
                        epoch=epoch,
                        g_optimizer=g_optimizer,
                        d_optimizer=d_optimizer,
                    )
                    
                    # 1. Log for MLflow
                    for k, v in model.loss_log.items():
                        mlflow.log_metric(k, v / len(train_loader), step=epoch)
                    
                    # 2. Log for WandB
                    wandb_metrics = {k: v / len(train_loader) for k, v in model.loss_log.items()}
                    wandb_metrics["epoch"] = epoch
                    
                    wandb.log(wandb_metrics)

                    logger.success(f"Epoch {epoch + 1} completed")

                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {str(e)}")
                    wandb.finish() 
                    raise

            logger.info("=" * 80)
            logger.success("Training completed successfully!")
            logger.info(f"Checkpoints saved to: {checkpoint_dir}")
            logger.info("=" * 80)
            
            # End Wandb run
            wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
