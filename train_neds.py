#!/usr/bin/env python3
"""
CONDSAR NeDS Training Script
Main entry point for training the ControlNet-based RGB-to-SAR generation model

Usage:
    # Stage A: Train on source domain
    python train_neds.py --config config_neds.yaml --stage a

    # Resume from checkpoint
    python train_neds.py --config config_neds.yaml --stage a --resume checkpoints/stage_a_neds/checkpoint_epoch_50.pt

    # Quick test with small dataset
    python train_neds.py --config config_neds.yaml --stage a --batch_size 2 --num_epochs 5
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

import argparse
import yaml
import logging
import torch

from models.training_stage_a_neds import StageATrainerNeDS
from models.training_utils import setup_logger

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def merge_args_with_config(config: dict, args: argparse.Namespace) -> dict:
    """Merge command-line arguments with config file (args take precedence)"""

    # Override config with command-line args
    if args.stage:
        config['stage'] = args.stage

    if args.source_dir:
        config['data']['source_dir'] = args.source_dir

    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    if args.num_epochs:
        config['training']['stage_a']['num_epochs'] = args.num_epochs

    if args.learning_rate:
        config['training']['stage_a']['learning_rate'] = args.learning_rate

    if args.device:
        config['hardware']['device'] = args.device

    if args.checkpoint_dir:
        config['training']['stage_a']['checkpoint_dir'] = args.checkpoint_dir

    if args.no_wandb:
        config['logging']['use_wandb'] = False

    return config


def validate_data_structure(data_dir: str) -> bool:
    """Validate that data directory has correct structure"""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"❌ Data directory not found: {data_path}")
        return False

    # Check for metadata.json
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"❌ metadata.json not found in {data_path}")
        logger.error("   Run generate_metadata_neds.py first to create metadata.json")
        return False

    # Check subdirectories
    pre_dir = data_path / "pre"
    post_dir = data_path / "post"
    mask_dir = data_path / "mask"

    missing_dirs = []
    if not pre_dir.exists():
        missing_dirs.append("pre/")
    if not post_dir.exists():
        missing_dirs.append("post/")
    if not mask_dir.exists():
        missing_dirs.append("mask/")

    if missing_dirs:
        logger.error(f"❌ Missing required directories: {', '.join(missing_dirs)}")
        logger.error(f"   Expected structure:")
        logger.error(f"   {data_path}/")
        logger.error(f"   ├── metadata.json")
        logger.error(f"   ├── pre/")
        logger.error(f"   ├── post/")
        logger.error(f"   └── mask/")
        return False

    logger.info(f"✅ Data structure validation passed: {data_path}")
    return True


def train_stage_a(config: dict, args: argparse.Namespace):
    """Train Stage A: ControlNet on source domain"""
    logger.info("=" * 80)
    logger.info("CONDSAR NeDS - Stage A Training")
    logger.info("=" * 80)

    # Extract config
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {}).get('stage_a', {})
    hardware_config = config.get('hardware', {})
    logging_config = config.get('logging', {})

    source_dir = data_config.get('source_dir', './condsar/data')

    # Validate data structure
    if not validate_data_structure(source_dir):
        logger.error("Data validation failed. Please fix the issues above.")
        sys.exit(1)

    # Initialize trainer
    trainer = StageATrainerNeDS(
        source_dataset_dir=source_dir,
        pretrained_model_name=model_config.get('pretrained_model_name', 'stabilityai/stable-diffusion-2-1-base'),
        num_disaster_types=data_config.get('num_disaster_types', 5),
        batch_size=data_config.get('batch_size', 4),
        num_epochs=training_config.get('num_epochs', 100),
        learning_rate=training_config.get('learning_rate', 1e-4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        device=hardware_config.get('device', 'cuda:0'),
        checkpoint_dir=training_config.get('checkpoint_dir', './checkpoints/stage_a_neds'),
        log_dir=logging_config.get('log_dir', './logs'),
        use_wandb=logging_config.get('use_wandb', True),
        wandb_project=logging_config.get('wandb_project', 'condsar_neds'),
        use_mixed_precision=training_config.get('use_mixed_precision', True),
        num_workers=data_config.get('num_workers', 4),
        save_every_n_epochs=training_config.get('save_every_n_epochs', 10),
    )

    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="CONDSAR NeDS Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Stage A with default config
  python train_neds.py --config config_neds.yaml --stage a
  
  # Train with custom settings
  python train_neds.py --config config_neds.yaml --stage a --batch_size 8 --num_epochs 200
  
  # Resume from checkpoint
  python train_neds.py --config config_neds.yaml --stage a --resume checkpoints/stage_a_neds/checkpoint_epoch_50.pt
  
  # Quick test (no W&B)
  python train_neds.py --config config_neds.yaml --stage a --no_wandb --num_epochs 5
        """
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=['a', 'b', 'c'],
        default='a',
        help="Training stage (a: source training, b: synthetic generation, c: mixed training)"
    )

    # Data arguments
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Source dataset directory (overrides config)"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (overrides config)"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs (overrides config)"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate (overrides config)"
    )

    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        help="Training device (cuda:0, cuda:1, cpu)"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Checkpoint directory (overrides config)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint path"
    )

    # Logging arguments
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(
        "condsar_neds",
        log_dir="./logs",
        level=logging.INFO
    )

    # Load and merge config
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = merge_args_with_config(config, args)

    # Print configuration summary
    logger.info("\nConfiguration Summary:")
    logger.info(f"  Stage: {config['stage']}")
    logger.info(f"  Source Dir: {config['data']['source_dir']}")
    logger.info(f"  Batch Size: {config['data']['batch_size']}")
    logger.info(f"  Num Epochs: {config['training']['stage_a']['num_epochs']}")
    logger.info(f"  Learning Rate: {config['training']['stage_a']['learning_rate']}")
    logger.info(f"  Device: {config['hardware']['device']}")
    logger.info(f"  Mixed Precision: {config['training']['stage_a']['use_mixed_precision']}")
    logger.info(f"  Use W&B: {config['logging']['use_wandb']}")
    logger.info("")

    # Check CUDA availability
    if 'cuda' in config['hardware']['device']:
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, falling back to CPU")
            config['hardware']['device'] = 'cpu'
        else:
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")

    # Train based on stage
    stage = config['stage']

    if stage == 'a':
        train_stage_a(config, args)
    elif stage == 'b':
        logger.error("Stage B not yet implemented")
        sys.exit(1)
    elif stage == 'c':
        logger.error("Stage C not yet implemented")
        sys.exit(1)
    else:
        logger.error(f"Unknown stage: {stage}")
        sys.exit(1)


if __name__ == "__main__":
    main()

