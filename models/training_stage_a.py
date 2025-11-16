"""
Stage A Training: Train ControlNet on source domain
Input: RGB pre + SAR post + building mask + disaster type
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Import diffusers components with fallback
try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler
except ImportError:
    try:
        from diffusers.pipelines import StableDiffusionPipeline
        from diffusers.schedulers import DDPMScheduler
    except ImportError:
        StableDiffusionPipeline = None
        DDPMScheduler = None

try:
    from diffusers.optimization import get_scheduler
except ImportError:
    try:
        from diffusers.utils import get_scheduler
    except ImportError:
        get_scheduler = None
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

try:
    from .enhanced_condsar import create_enhanced_controlnet
    from .training_utils import (
        setup_logger, setup_wandb, DisasterSARDataset,
        MetricsTracker, save_checkpoint, load_checkpoint,
        log_to_wandb
    )
    from .weighted_sampler import WeightedSamplerConfig, create_weighted_dataloader
except ImportError:
    # Fallback for direct imports
    from enhanced_condsar import create_enhanced_controlnet
    from training_utils import (
        setup_logger, setup_wandb, DisasterSARDataset,
        MetricsTracker, save_checkpoint, load_checkpoint,
        log_to_wandb
    )
    from weighted_sampler import WeightedSamplerConfig, create_weighted_dataloader

logger = logging.getLogger(__name__)


class StageATrainer:
    """Stage A: Train ControlNet on source domain"""

    def __init__(
        self,
        source_dataset_dir: str,
        pretrained_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        num_disaster_types: int = 5,
        batch_size: int = 4,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda:0",
        checkpoint_dir: str = "./checkpoints/stage_a",
        log_dir: str = "./logs",
        wandb_config: Optional[Dict[str, Any]] = None,
        use_mixed_precision: bool = True,
        num_workers: int = 4,
        use_weighted_sampler: bool = True,
        weight_strategy: str = "inverse_frequency",
        weight_temperature: float = 1.0,
    ):
        """
        Initialize Stage A trainer

        Args:
            source_dataset_dir: Path to source domain dataset
            pretrained_model_name: Base diffusion model name
            num_disaster_types: Number of disaster types
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            wandb_config: W&B configuration
            use_mixed_precision: Whether to use mixed precision training
            num_workers: Number of data loader workers
            use_weighted_sampler: Whether to use weighted sampling for imbalanced classes
            weight_strategy: Weight calculation strategy ("inverse_frequency", "sqrt_frequency", "custom", "balanced")
            weight_temperature: Temperature for weight smoothing (higher = more balanced)
        """
        self.device = torch.device(device)
        self.num_disaster_types = num_disaster_types
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.num_workers = num_workers

        # Weighted sampler parameters
        self.use_weighted_sampler = use_weighted_sampler
        self.weight_strategy = weight_strategy
        self.weight_temperature = weight_temperature

        # Setup logging
        self.logger = setup_logger(
            "stage_a_trainer",
            log_dir=log_dir,
            level=logging.INFO
        )

        # Setup W&B
        wandb_config = wandb_config or {}
        wandb_config.update({
            "stage": "A",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_disaster_types": num_disaster_types,
            "use_mixed_precision": use_mixed_precision,
            "use_weighted_sampler": use_weighted_sampler,
            "weight_strategy": weight_strategy,
            "weight_temperature": weight_temperature,
        })
        self.wandb_run = setup_wandb(
            project_name="condsar_stage_a",
            run_name="stage_a_training",
            config=wandb_config
        )

        self.logger.info("=" * 80)
        self.logger.info("Stage A Training: Source Domain ControlNet Training")
        self.logger.info("=" * 80)

        # Load dataset
        self.logger.info(f"Loading source dataset from {source_dataset_dir}")
        self.train_dataset = DisasterSARDataset(
            dataset_dir=source_dataset_dir,
            image_size=512,
            return_mask=True,
            return_metadata=True,
            logger=self.logger
        )

        # Create data loader with optional weighted sampling
        if use_weighted_sampler:
            self.logger.info(f"Using weighted sampler: strategy={weight_strategy}, temperature={weight_temperature}")
            sampler_config = WeightedSamplerConfig(
                use_weighted_sampler=True,
                weight_strategy=weight_strategy,
                temperature=weight_temperature,
                replacement=True
            )
            self.train_loader = create_weighted_dataloader(
                dataset=self.train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler_config=sampler_config
            )
        else:
            self.logger.info("Using standard sampler (no weighting)")
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )

        self.logger.info(f"Dataset loaded: {len(self.train_dataset)} samples")
        self.logger.info(f"Data loader: {len(self.train_loader)} batches")

        # Load pretrained models
        self.logger.info(f"Loading base diffusion model: {pretrained_model_name}")

        # Load VAE
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Load UNet
        from diffusers import UNet2DConditionModel
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.float32,
        ).to(self.device)
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False

        # Load ControlNet
        self.logger.info("Creating EnhancedDisasterControlNet")
        self.controlnet = create_enhanced_controlnet(
            pretrained_model_name_or_path=pretrained_model_name,
            num_disaster_types=num_disaster_types,
            disaster_embedding_dim=128,
            num_severity_levels=4,
            severity_embedding_dim=128,
            torch_dtype=torch.float32
        ).to(self.device)
        self.controlnet.train()

        # Only train ControlNet parameters
        self.trainable_params = list(self.controlnet.parameters())

        # Setup optimizer
        self.optimizer = AdamW(
            self.trainable_params,
            lr=learning_rate,
            weight_decay=1e-2,
            eps=1e-8
        )

        # Setup scheduler
        self.num_training_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=self.num_training_steps,
        )

        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler"
        )

        # Setup mixed precision training
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Metrics tracker
        self.metrics = MetricsTracker(logger=self.logger)

        # Checkpointing
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.logger.info("Stage A trainer initialized successfully")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.metrics.reset()
        self.controlnet.train()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            rgb_image = batch['rgb_image'].to(self.device)
            sar_image = batch['sar_image'].to(self.device)
            building_mask = batch['building_mask'].to(self.device) if batch['building_mask'] is not None else None
            disaster_type = batch['disaster_type'].to(self.device) if batch['disaster_type'] is not None else None
            disaster_severity = batch['disaster_severity'].to(self.device) if batch['disaster_severity'] is not None else None

            # Encode SAR image to latent space
            with torch.no_grad():
                # Scale RGB and SAR to [-1, 1]
                rgb_image = rgb_image * 2 - 1
                sar_image = sar_image * 2 - 1

                # Encode SAR (post-disaster) to latent
                z_sar = self.vae.encode(sar_image).latent_dist.sample()
                z_sar = z_sar * self.vae.config.scaling_factor

            # Sample random timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (rgb_image.size(0),),
                device=self.device
            )

            # Sample noise
            noise = torch.randn_like(z_sar)

            # Add noise to latents (forward diffusion)
            z_t = self.noise_scheduler.add_noise(z_sar, noise, timesteps)

            # Forward pass through ControlNet and UNet
            with torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext():
                # Get control embeddings from ControlNet
                control_output = self.controlnet(
                    sample=z_t,
                    timestep=timesteps,
                    encoder_hidden_states=None,
                    rgb_image=rgb_image,
                    building_mask=building_mask,
                    disaster_type=disaster_type,
                    disaster_severity=disaster_severity,
                    return_dict=True
                )

                # Predict noise with UNet using control embeddings
                pred_noise = self.unet(
                    sample=z_t,
                    timestep=timesteps,
                    encoder_hidden_states=None,
                    down_block_additional_residuals=control_output.down_block_res_samples,
                    mid_block_additional_residual=control_output.mid_block_res_sample,
                ).sample

            # Compute loss (simple MSE loss for now)
            loss = F.mse_loss(pred_noise, noise)
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            # Log metrics
            self.metrics.update(loss=loss.item() * self.gradient_accumulation_steps)

            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item() * self.gradient_accumulation_steps:.6f}"
                )

        # Log epoch metrics
        avg_loss = self.metrics.get_mean('loss')
        self.metrics.log_epoch(epoch, lr=self.optimizer.param_groups[0]['lr'])

        # Log to W&B
        log_to_wandb({
            'epoch': epoch,
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })

        return avg_loss

    def train(self):
        """Train for multiple epochs"""
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.logger.info(f"{'='*60}")

            avg_loss = self.train_epoch(epoch)

            # Save checkpoint
            save_checkpoint(
                self.controlnet,
                self.optimizer,
                epoch,
                save_dir=self.checkpoint_dir,
                is_best=(avg_loss < best_loss),
                logger=self.logger
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.logger.info(f"New best loss: {best_loss:.6f}")

        self.logger.info("\n" + "="*80)
        self.logger.info("Stage A Training Completed!")
        self.logger.info("="*80)

        if self.wandb_run is not None:
            self.wandb_run.finish()


from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser(description="Stage A Training: Source Domain ControlNet")
    parser.add_argument("--source_dataset_dir", type=str, required=True, help="Path to source dataset")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage_a")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--use_mixed_precision", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_disabled", action="store_true")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("stage_a_main", log_dir=args.log_dir)
    logger.info(f"Arguments: {args}")

    # Create trainer
    trainer = StageATrainer(
        source_dataset_dir=args.source_dataset_dir,
        pretrained_model_name=args.pretrained_model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_mixed_precision=args.use_mixed_precision,
        num_workers=args.num_workers,
        wandb_config={"wandb_offline": args.wandb_offline}
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

