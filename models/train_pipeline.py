"""
CONDSAR Training Pipeline - Unified configuration and launcher
Supports Stage A, B, and C training in sequence or individually
"""
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from training_utils import setup_logger, setup_wandb

logger = logging.getLogger(__name__)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for all stages"""
    return {
        "project_name": "condsar",
        "seed": 42,

        # Stage A Configuration
        "stage_a": {
            "enabled": True,
            "source_dataset_dir": "./data",
            "pretrained_model": "stabilityai/stable-diffusion-2-1-base",
            "batch_size": 4,
            "num_epochs": 100,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 1,
            "num_workers": 4,
            "use_mixed_precision": True,
            "checkpoint_dir": "./checkpoints/stage_a",
            "log_dir": "./logs",
        },

        # Stage B Configuration
        "stage_b": {
            "enabled": True,
            "target_dataset_dir": "./data",
            "controlnet_model_path": "./checkpoints/stage_a/best_model.pt",
            "pretrained_model": "stabilityai/stable-diffusion-2-1-base",
            "batch_size": 4,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
            "num_workers": 4,
            "output_dir": "./synthetic_data",
            "log_dir": "./logs",
            "num_variants_per_sample": 1,
        },

        # Stage C Configuration
        "stage_c": {
            "enabled": True,
            "source_dataset_dir": "./data",
            "synthetic_dataset_dir": "./synthetic_data",
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 1e-3,
            "num_workers": 4,
            "synthetic_weight": 0.5,
            "checkpoint_dir": "./checkpoints/stage_c",
            "log_dir": "./logs",
        },

        # Common Configuration
        "device": "cuda:0",
        "wandb_offline": False,
        "wandb_disabled": False,
    }


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to JSON file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded configuration from {path}")
    return config


def train_stage_a(config: Dict[str, Any]):
    """Run Stage A training"""
    from training_stage_a import StageATrainer

    logger.info("\n" + "=" * 80)
    logger.info("STARTING STAGE A TRAINING")
    logger.info("=" * 80)

    cfg = config['stage_a']

    trainer = StageATrainer(
        source_dataset_dir=cfg['source_dataset_dir'],
        pretrained_model_name=cfg['pretrained_model'],
        num_disaster_types=5,
        batch_size=cfg['batch_size'],
        num_epochs=cfg['num_epochs'],
        learning_rate=cfg['learning_rate'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        device=config['device'],
        checkpoint_dir=cfg['checkpoint_dir'],
        log_dir=cfg['log_dir'],
        wandb_config={'wandb_offline': config['wandb_offline']},
        use_mixed_precision=cfg['use_mixed_precision'],
        num_workers=cfg['num_workers'],
    )

    trainer.train()


def train_stage_b(config: Dict[str, Any]):
    """Run Stage B generation"""
    from training_stage_b import StageBGenerator

    logger.info("\n" + "=" * 80)
    logger.info("STARTING STAGE B GENERATION")
    logger.info("=" * 80)

    cfg = config['stage_b']

    generator = StageBGenerator(
        target_dataset_dir=cfg['target_dataset_dir'],
        controlnet_model_path=cfg['controlnet_model_path'],
        pretrained_model_name=cfg['pretrained_model'],
        num_disaster_types=5,
        batch_size=cfg['batch_size'],
        device=config['device'],
        output_dir=cfg['output_dir'],
        log_dir=cfg['log_dir'],
        num_inference_steps=cfg['num_inference_steps'],
        guidance_scale=cfg['guidance_scale'],
        controlnet_conditioning_scale=cfg['controlnet_conditioning_scale'],
        seed=config['seed'],
        num_workers=cfg['num_workers'],
    )

    generator.generate_all(
        num_variants_per_sample=cfg['num_variants_per_sample']
    )


def train_stage_c(config: Dict[str, Any]):
    """Run Stage C mixed training"""
    from training_stage_c import StageCTrainer

    logger.info("\n" + "=" * 80)
    logger.info("STARTING STAGE C MIXED TRAINING")
    logger.info("=" * 80)

    cfg = config['stage_c']

    trainer = StageCTrainer(
        source_dataset_dir=cfg['source_dataset_dir'],
        synthetic_dataset_dir=cfg['synthetic_dataset_dir'],
        batch_size=cfg['batch_size'],
        num_epochs=cfg['num_epochs'],
        learning_rate=cfg['learning_rate'],
        device=config['device'],
        checkpoint_dir=cfg['checkpoint_dir'],
        log_dir=cfg['log_dir'],
        num_workers=cfg['num_workers'],
        synthetic_weight=cfg['synthetic_weight'],
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="CONDSAR Training Pipeline - Disaster-Aware SAR Image Generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file (if not provided, uses defaults)"
    )
    parser.add_argument(
        "--save_default_config",
        type=str,
        default=None,
        help="Save default configuration to file and exit"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["a", "b", "c", "all"],
        default="all",
        help="Which stage(s) to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--source_dataset_dir",
        type=str,
        default=None,
        help="Override source dataset directory"
    )
    parser.add_argument(
        "--target_dataset_dir",
        type=str,
        default=None,
        help="Override target dataset directory"
    )
    parser.add_argument(
        "--synthetic_data_dir",
        type=str,
        default=None,
        help="Override synthetic data directory"
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run W&B in offline mode"
    )
    parser.add_argument(
        "--wandb_disabled",
        action="store_true",
        help="Disable W&B entirely"
    )

    args = parser.parse_args()

    # Setup logging
    base_logger = setup_logger("condsar_pipeline", level=logging.INFO)

    base_logger.info("CONDSAR Training Pipeline")
    base_logger.info("=" * 80)

    # Handle config saving
    if args.save_default_config:
        config = create_default_config()
        save_config(config, args.save_default_config)
        base_logger.info(f"Saved default configuration to {args.save_default_config}")
        sys.exit(0)

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        base_logger.info("Using default configuration")

    # Override settings from command line
    if args.device:
        config['device'] = args.device
    if args.source_dataset_dir:
        config['stage_a']['source_dataset_dir'] = args.source_dataset_dir
        config['stage_c']['source_dataset_dir'] = args.source_dataset_dir
    if args.target_dataset_dir:
        config['stage_b']['target_dataset_dir'] = args.target_dataset_dir
    if args.synthetic_data_dir:
        config['stage_b']['output_dir'] = args.synthetic_data_dir
        config['stage_c']['synthetic_dataset_dir'] = args.synthetic_data_dir
    if args.wandb_offline:
        config['wandb_offline'] = True
    if args.wandb_disabled:
        config['wandb_disabled'] = True

    base_logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Run stages
    try:
        if args.stage in ["a", "all"]:
            if config['stage_a']['enabled']:
                train_stage_a(config)
            else:
                base_logger.info("Stage A disabled in configuration")

        if args.stage in ["b", "all"]:
            if config['stage_b']['enabled']:
                train_stage_b(config)
            else:
                base_logger.info("Stage B disabled in configuration")

        if args.stage in ["c", "all"]:
            if config['stage_c']['enabled']:
                train_stage_c(config)
            else:
                base_logger.info("Stage C disabled in configuration")

        base_logger.info("\n" + "=" * 80)
        base_logger.info("All specified stages completed successfully!")
        base_logger.info("=" * 80)

    except Exception as e:
        base_logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

