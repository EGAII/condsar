"""
Stage A Training: Train CONDSAR NeDS on Source Domain
Based on NeDS architecture and training procedure

Training Loop (following NeDS pseudo-code):
1. Load pre-event RGB image, post-event SAR (GT), mask, disaster type/severity
2. Encode pre-event RGB via frozen VAE encoder -> pre_latents
3. Add noise to pre_latents at timestep t -> noisy_latents
4. Pass through ControlNet with mask + disaster embeddings
5. Predict noise and compute diffusion loss
6. Backprop and update (only ControlNet + SAR decoder, VAE frozen)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import wandb

from condsar_neds import CONDSARNeDS, SARVAEDecoder, create_condsar_neds_from_pretrained
from training_utils import (
    DisasterSARDataset,
    MetricsTracker,
    save_checkpoint,
    load_checkpoint,
    setup_logger
)

logger = logging.getLogger(__name__)


class StageATrainerNeDS:
    """
    Stage A Trainer for CONDSAR NeDS

    Trains ControlNet on source domain with:
    - Input: Pre-event RGB (via frozen VAE)
    - Condition: Building/damage mask
    - Embeddings: Disaster type + severity
    - Ground Truth: Post-event SAR image
    """

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
        checkpoint_dir: str = "./checkpoints/stage_a_neds",
        log_dir: str = "./logs",
        use_wandb: bool = True,
        wandb_project: str = "condsar_neds_stage_a",
        use_mixed_precision: bool = True,
        num_workers: int = 4,
        save_every_n_epochs: int = 10,
    ):
        """
        Initialize Stage A trainer for NeDS-based CONDSAR

        Args:
            source_dataset_dir: Path to source domain dataset
            pretrained_model_name: Base SD model name
            num_disaster_types: Number of disaster types (default: 5)
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            device: Training device
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            use_mixed_precision: Whether to use FP16 mixed precision
            num_workers: DataLoader workers
            save_every_n_epochs: Save checkpoint every N epochs
        """
        self.device = torch.device(device)
        self.num_disaster_types = num_disaster_types
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.num_workers = num_workers
        self.save_every_n_epochs = save_every_n_epochs
        self.use_wandb = use_wandb

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            "stage_a_neds_trainer",
            log_dir=str(self.log_dir),
            level=logging.INFO
        )

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "stage": "A",
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "num_disaster_types": num_disaster_types,
                    "use_mixed_precision": use_mixed_precision,
                    "model": "CONDSAR_NeDS",
                }
            )

        self.logger.info("=" * 80)
        self.logger.info("CONDSAR NeDS - Stage A Training")
        self.logger.info("=" * 80)

        # ========== Load Dataset ==========
        self.logger.info(f"Loading source dataset from {source_dataset_dir}")
        self.train_dataset = DisasterSARDataset(
            dataset_dir=source_dataset_dir,
            image_size=512,
            return_mask=True,
            return_metadata=True,
            logger=self.logger
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.logger.info(f"Dataset loaded: {len(self.train_dataset)} samples")

        # ========== Load Models ==========
        self.logger.info("Loading models...")

        # 1. VAE (frozen for encoding RGB to latent)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.logger.info("✅ VAE loaded and frozen")

        # 2. UNet (for diffusion, frozen during Stage A)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.float32
        ).to(self.device)
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False
        self.logger.info("✅ UNet loaded and frozen")

        # 3. ControlNet (CONDSAR NeDS - trainable)
        self.controlnet = create_condsar_neds_from_pretrained(
            pretrained_model_name=pretrained_model_name,
            num_disaster_types=num_disaster_types,
            conditioning_channels=1,  # Single-channel mask
            torch_dtype=torch.float32,
        ).to(self.device)
        self.controlnet.train()
        self.logger.info("✅ CONDSAR NeDS ControlNet initialized")

        # 4. SAR VAE Decoder (trainable)
        self.sar_decoder = SARVAEDecoder(
            latent_channels=4,
            latent_size=64,
            output_channels=1,
            hidden_channels=128
        ).to(self.device)
        self.sar_decoder.train()
        self.logger.info("✅ SAR VAE Decoder initialized")

        # 5. Noise scheduler (DDPM for training)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler"
        )
        self.logger.info("✅ Noise scheduler loaded")

        # ========== Optimizer & Scheduler ==========
        trainable_params = list(self.controlnet.parameters()) + list(self.sar_decoder.parameters())
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_loader) * num_epochs,
        )

        self.logger.info(f"✅ Optimizer: AdamW, LR={learning_rate}")
        self.logger.info(f"✅ Scheduler: Cosine with 500 warmup steps")

        # ========== Mixed Precision ==========
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        if use_mixed_precision:
            self.logger.info("✅ Mixed precision training enabled")

        # ========== Metrics Tracker ==========
        self.metrics = MetricsTracker()

        self.logger.info("=" * 80)
        self.logger.info("Stage A Trainer initialized successfully")
        self.logger.info("=" * 80)

    def train(self):
        """Main training loop following NeDS Stage A procedure"""
        self.logger.info("Starting Stage A training...")

        global_step = 0

        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            epoch_loss = 0.0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                rgb_image = batch['rgb_image'].to(self.device)  # (B, 3, 512, 512)
                sar_gt = batch['sar_image'].to(self.device)     # (B, 1, 512, 512)
                mask = batch['building_mask'].to(self.device)    # (B, 1, 512, 512)
                disaster_type = batch['disaster_type'].to(self.device)  # (B,)
                disaster_severity = batch['disaster_severity'].to(self.device)  # (B,)

                # ========== NeDS Training Step ==========
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    # 1. Encode pre-event RGB via frozen VAE
                    with torch.no_grad():
                        # Convert RGB [0,1] to [-1,1] for VAE
                        rgb_norm = rgb_image * 2.0 - 1.0
                        pre_latents = self.vae.encode(rgb_norm).latent_dist.sample()
                        pre_latents = pre_latents * self.vae.config.scaling_factor

                    # 2. Add noise to pre_latents (diffusion forward process)
                    noise = torch.randn_like(pre_latents)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (pre_latents.shape[0],),
                        device=self.device
                    ).long()

                    noisy_latents = self.noise_scheduler.add_noise(
                        pre_latents,
                        noise,
                        timesteps
                    )

                    # 3. Prepare encoder hidden states (dummy for now, can use CLIP text embeddings)
                    encoder_hidden_states = torch.zeros(
                        rgb_image.shape[0],
                        77,  # CLIP sequence length
                        1024  # SD2.1 text embedding dim
                    ).to(self.device)

                    # 4. ControlNet forward pass
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        pre_latents=pre_latents,
                        mask=mask,
                        disaster_type=disaster_type,
                        disaster_severity=disaster_severity,
                        return_dict=False,
                    )

                    # 5. UNet prediction (with ControlNet conditioning)
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # 6. Compute diffusion loss (MSE between predicted and true noise)
                    diffusion_loss = F.mse_loss(noise_pred, noise, reduction="mean")

                    # 7. Optional: Add reconstruction loss via SAR decoder
                    # Decode predicted clean latent to SAR image
                    with torch.no_grad():
                        # Predict clean latent (x0)
                        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps]
                        beta_prod_t = 1 - alpha_prod_t
                        pred_original_sample = (
                            noisy_latents - beta_prod_t.sqrt().view(-1, 1, 1, 1) * noise_pred
                        ) / alpha_prod_t.sqrt().view(-1, 1, 1, 1)

                    # Decode to SAR image
                    sar_pred = self.sar_decoder(pred_original_sample)

                    # Normalize SAR GT to [-1, 1] to match decoder output
                    sar_gt_norm = sar_gt * 2.0 - 1.0

                    # Reconstruction loss
                    recon_loss = F.l1_loss(sar_pred, sar_gt_norm)

                    # Total loss
                    loss = diffusion_loss + 0.1 * recon_loss  # Weight recon loss lower

                # ========== Backward Pass ==========
                loss = loss / self.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # ========== Logging ==========
                epoch_loss += loss.item() * self.gradient_accumulation_steps

                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'diff': f'{diffusion_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                })

                if self.use_wandb and global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss.item() * self.gradient_accumulation_steps,
                        'train/diffusion_loss': diffusion_loss.item(),
                        'train/reconstruction_loss': recon_loss.item(),
                        'train/lr': self.lr_scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'global_step': global_step,
                    })

            # ========== Epoch Summary ==========
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss:.4f}")

            # ========== Save Checkpoint ==========
            if (epoch + 1) % self.save_every_n_epochs == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, global_step)
                self.logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Stage A training completed!")
        self.logger.info("=" * 80)

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        self.save_checkpoint(final_path, self.num_epochs - 1, global_step)
        self.logger.info(f"✅ Final model saved: {final_path}")

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, path: Path, epoch: int, global_step: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'controlnet_state_dict': self.controlnet.state_dict(),
            'sar_decoder_state_dict': self.sar_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.controlnet.load_state_dict(checkpoint['controlnet_state_dict'])
        self.sar_decoder.load_state_dict(checkpoint['sar_decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"✅ Loaded checkpoint from {path}")
        return checkpoint['epoch'], checkpoint['global_step']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CONDSAR NeDS Stage A")
    parser.add_argument("--source_dir", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--num_disaster_types", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage_a_neds")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    trainer = StageATrainerNeDS(
        source_dataset_dir=args.source_dir,
        pretrained_model_name=args.pretrained_model,
        num_disaster_types=args.num_disaster_types,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
    )

    trainer.train()

