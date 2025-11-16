"""
Stage C: Mixed training on downstream tasks
Input: Real data (source) + Synthetic data (target)
Task: Building damage classification and localization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import argparse
import json

from training_utils import (
    setup_logger, setup_wandb, DisasterSARDataset,
    MetricsTracker, save_checkpoint, load_checkpoint,
    log_to_wandb
)

logger = logging.getLogger(__name__)


class BuildingDamageClassifier(nn.Module):
    """Simple CNN-based building damage classifier"""

    def __init__(self, num_classes: int = 4):
        """
        Args:
            num_classes: Number of damage classes (0=Intact, 1=Minor, 2=Major, 3=Destroyed)
        """
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            # Input: 1 channel (SAR) + 3 channels (RGB) = 4 channels
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, sar_image: torch.Tensor, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar_image: SAR image (B, 1, H, W)
            rgb_image: RGB image (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        # Concatenate SAR and RGB
        x = torch.cat([sar_image, rgb_image], dim=1)  # (B, 4, H, W)

        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # (B, 256)

        # Classify
        logits = self.classifier(features)
        return logits


class StageCTrainer:
    """Stage C: Mixed training on downstream tasks"""

    def __init__(
        self,
        source_dataset_dir: str,
        synthetic_dataset_dir: str,
        batch_size: int = 16,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = "cuda:0",
        checkpoint_dir: str = "./checkpoints/stage_c",
        log_dir: str = "./logs",
        num_workers: int = 4,
        synthetic_weight: float = 0.5,
    ):
        """
        Initialize Stage C trainer

        Args:
            source_dataset_dir: Path to source domain dataset
            synthetic_dataset_dir: Path to synthetic dataset
            batch_size: Batch size per domain
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            num_workers: Number of data loader workers
            synthetic_weight: Weight for synthetic data loss
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.synthetic_weight = synthetic_weight

        # Setup logging
        self.logger = setup_logger(
            "stage_c_trainer",
            log_dir=log_dir,
            level=logging.INFO
        )

        # Setup W&B
        wandb_config = {
            "stage": "C",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "synthetic_weight": synthetic_weight,
        }
        self.wandb_run = setup_wandb(
            project_name="condsar_stage_c",
            run_name="stage_c_mixed_training",
            config=wandb_config
        )

        self.logger.info("=" * 80)
        self.logger.info("Stage C: Mixed Training on Downstream Tasks")
        self.logger.info("=" * 80)

        # Load source dataset
        self.logger.info(f"Loading source dataset from {source_dataset_dir}")
        self.source_dataset = DisasterSARDataset(
            dataset_dir=source_dataset_dir,
            image_size=512,
            return_mask=True,
            return_metadata=True,
            logger=self.logger
        )

        self.source_loader = DataLoader(
            self.source_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.logger.info(f"Source dataset: {len(self.source_dataset)} samples")

        # Load synthetic dataset
        self.logger.info(f"Loading synthetic dataset from {synthetic_dataset_dir}")
        self.synthetic_dataset = SyntheticSARDataset(
            dataset_dir=synthetic_dataset_dir,
            image_size=512,
            logger=self.logger
        )

        self.synthetic_loader = DataLoader(
            self.synthetic_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.logger.info(f"Synthetic dataset: {len(self.synthetic_dataset)} samples")

        # Create model
        self.logger.info("Creating damage classifier model")
        self.model = BuildingDamageClassifier(num_classes=4).to(self.device)
        self.model.train()

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-2
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()

        # Metrics tracker
        self.metrics = MetricsTracker(logger=self.logger)

        # Checkpointing
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.logger.info("Stage C trainer initialized successfully")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch on mixed data"""
        self.metrics.reset()
        self.model.train()

        # Create iterators for both loaders
        source_iter = iter(self.source_loader)
        synthetic_iter = iter(self.synthetic_loader)

        num_batches = max(len(self.source_loader), len(self.synthetic_loader))

        for batch_idx in range(num_batches):
            # Get source batch
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_loader)
                source_batch = next(source_iter)

            # Get synthetic batch
            try:
                synthetic_batch = next(synthetic_iter)
            except StopIteration:
                synthetic_iter = iter(self.synthetic_loader)
                synthetic_batch = next(synthetic_iter)

            # ========== Source Domain Loss ==========
            rgb_src = source_batch['rgb_image'].to(self.device)
            sar_src = source_batch['sar_image'].to(self.device)

            # Get damage labels from metadata (if available)
            # For now, use placeholder labels from disaster type + severity
            if 'disaster_type' in source_batch:
                # Simple heuristic: map disaster type and severity to damage level
                disaster_type = source_batch['disaster_type'].to(self.device)
                severity = source_batch['disaster_severity'].to(self.device)

                # Damage level = (disaster_type + 1) * severity -> [0, 3]
                damage_labels = (
                    ((disaster_type.float() + 1) * severity * 1.5).long()
                ).clamp(0, 3).to(self.device)
            else:
                # Use random labels if no metadata
                damage_labels = torch.randint(0, 4, (sar_src.size(0),), device=self.device)

            # Forward pass
            pred_src = self.model(sar_src, rgb_src)
            loss_src = self.ce_loss(pred_src, damage_labels)

            # ========== Synthetic Domain Loss ==========
            sar_syn = synthetic_batch['sar_image'].to(self.device)
            rgb_syn = synthetic_batch['rgb_image'].to(self.device)
            damage_labels_syn = synthetic_batch['damage_label'].to(self.device)

            # Forward pass (only use classification loss for synthetic)
            pred_syn = self.model(sar_syn, rgb_syn)
            loss_syn = self.ce_loss(pred_syn, damage_labels_syn)

            # ========== Combined Loss ==========
            total_loss = loss_src + self.synthetic_weight * loss_syn

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Calculate accuracy
            acc_src = (pred_src.argmax(dim=1) == damage_labels).float().mean()
            acc_syn = (pred_syn.argmax(dim=1) == damage_labels_syn).float().mean()

            # Log metrics
            self.metrics.update(
                loss_src=loss_src.item(),
                loss_syn=loss_syn.item(),
                loss_total=total_loss.item(),
                acc_src=acc_src.item(),
                acc_syn=acc_syn.item(),
            )

            if batch_idx % 20 == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss_src: {loss_src.item():.4f}, "
                    f"Loss_syn: {loss_syn.item():.4f}, "
                    f"Acc_src: {acc_src.item():.4f}, "
                    f"Acc_syn: {acc_syn.item():.4f}"
                )

        # Log epoch metrics
        metrics_dict = self.metrics.to_dict()
        self.metrics.log_epoch(epoch, lr=self.optimizer.param_groups[0]['lr'])

        # Log to W&B
        log_to_wandb({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in metrics_dict.items()},
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })

        return metrics_dict

    def train(self):
        """Train for multiple epochs"""
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.logger.info(f"{'='*60}")

            metrics = self.train_epoch(epoch)
            self.scheduler.step()

            # Save checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                save_dir=self.checkpoint_dir,
                is_best=(metrics.get('acc_src', 0) > best_acc),
                logger=self.logger
            )

            if metrics.get('acc_src', 0) > best_acc:
                best_acc = metrics['acc_src']
                self.logger.info(f"New best accuracy: {best_acc:.4f}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Stage C Mixed Training Completed!")
        self.logger.info(f"Best accuracy: {best_acc:.4f}")
        self.logger.info("=" * 80)

        if self.wandb_run is not None:
            self.wandb_run.finish()


class SyntheticSARDataset:
    """Dataset for synthetic SAR images from Stage B"""

    def __init__(
        self,
        dataset_dir: str,
        image_size: int = 512,
        logger: logging.Logger = None
    ):
        """
        Args:
            dataset_dir: Path to synthetic dataset (output from Stage B)
            image_size: Image size
            logger: Logger instance
        """
        from pathlib import Path
        from PIL import Image
        import numpy as np

        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.logger = logger or logging.getLogger(__name__)

        # Load metadata
        metadata_path = self.dataset_dir / "synthetic_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.image_ids = list(self.metadata.keys())
        self.logger.info(f"Loaded {len(self.image_ids)} synthetic samples")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        img_id = self.image_ids[idx]
        meta = self.metadata[img_id]

        # Load SAR image
        sar_image = Image.open(meta['sar_path']).convert('L')
        sar_image = sar_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        sar_tensor = torch.from_numpy(np.array(sar_image)).unsqueeze(0).float() / 255.0

        # Load RGB image
        rgb_image = Image.open(meta['rgb_path']).convert('RGB')
        rgb_image = rgb_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0

        # Damage label from severity
        severity = meta['disaster_severity']
        damage_label = int(severity * 3)  # Convert [0, 1] to [0, 3]
        damage_label = min(damage_label, 3)
        damage_label_tensor = torch.tensor(damage_label, dtype=torch.long)

        return {
            'image_id': img_id,
            'sar_image': sar_tensor,
            'rgb_image': rgb_tensor,
            'damage_label': damage_label_tensor,
            'disaster_type': torch.tensor(meta['disaster_type'], dtype=torch.long),
            'disaster_severity': torch.tensor(meta['disaster_severity'], dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(description="Stage C: Mixed Training on Downstream Tasks")
    parser.add_argument("--source_dataset_dir", type=str, required=True, help="Path to source dataset")
    parser.add_argument("--synthetic_dataset_dir", type=str, required=True, help="Path to synthetic dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage_c")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--synthetic_weight", type=float, default=0.5)
    parser.add_argument("--wandb_offline", action="store_true")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("stage_c_main", log_dir=args.log_dir)
    logger.info(f"Arguments: {args}")

    # Create trainer
    trainer = StageCTrainer(
        source_dataset_dir=args.source_dataset_dir,
        synthetic_dataset_dir=args.synthetic_dataset_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_workers=args.num_workers,
        synthetic_weight=args.synthetic_weight,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

