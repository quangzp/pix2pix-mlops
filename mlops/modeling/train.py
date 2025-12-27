from pathlib import Path
from typing import Optional

from loguru import logger
import torch
import typer

from mlops.src.components.discriminator import define_D
from mlops.src.components.generator import define_G
from mlops.src.components.losses import GANLoss, VGGLoss
from mlops.src.components.replay_pool import ReplayPool
from mlops.src.models.pix2pixhd_module import Pix2PixHD, Pix2PixHDDataset

app = typer.Typer()


@app.command()
def main(
    # ---- Dataset paths ----
    dataset_path: Path = typer.Option(Path("./data/raw/face"), help="Root dataset directory"),
    feature_folder: str = typer.Option("/sketches/", help="Subfolder containing input images"),
    label_folder: str = typer.Option("/photos/", help="Subfolder containing target images"),
    # ---- Model configuration ----
    checkpoint_dir: Path = typer.Option(
        Path("./checkpoints"), help="Directory to save checkpoints"
    ),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Batch size for training"),
    # ---- Generator config ----
    ngf: int = typer.Option(64, help="Number of generator filters"),
    n_downsample_global: int = typer.Option(3, help="Number of global downsampling layers"),
    n_blocks_global: int = typer.Option(9, help="Number of global residual blocks"),
    n_local_enhancers: int = typer.Option(1, help="Number of local enhancers"),
    n_blocks_local: int = typer.Option(3, help="Number of local residual blocks"),
    # ---- Discriminator config ----
    ndf: int = typer.Option(64, help="Number of discriminator filters"),
    n_layers_D: int = typer.Option(3, help="Number of discriminator layers"),
    num_D: int = typer.Option(3, help="Number of discriminators (multi-scale)"),
    # ---- Training config ----
    learning_rate: float = typer.Option(1e-4, help="Learning rate for optimizers"),
    lambda_feat: float = typer.Option(10.0, help="Weight for feature matching loss"),
    replay_pool_size: int = typer.Option(50, help="Size of replay buffer"),
    ema_decay: float = typer.Option(0.9999, help="EMA decay rate for generator"),
    # ---- Data config ----
    img_size: int = typer.Option(256, help="Image size for training"),
    num_workers: int = typer.Option(4, help="Number of data loading workers"),
    # ---- Training control ----
    test_interval: int = typer.Option(100, help="Test image generation interval (iterations)"),
    save_interval: int = typer.Option(1000, help="Checkpoint save interval (iterations)"),
    resume_from: Optional[Path] = typer.Option(None, help="Path to checkpoint to resume from"),
):
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
                logger.success(f"Epoch {epoch + 1} completed")

            except Exception as e:
                logger.error(f"Error during epoch {epoch + 1}: {str(e)}")
                raise

        logger.info("=" * 80)
        logger.success("Training completed successfully!")
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    app()
