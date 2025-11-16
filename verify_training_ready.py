#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training script readiness check - no emojis"""
import sys
import warnings
from pathlib import Path

# Suppress Pydantic v2 compatibility warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Field.*has no effect.*')
warnings.filterwarnings('ignore', module='pydantic._internal._generate_schema')

sys.path.insert(0, str(Path(__file__).parent))

def test_training_ready():
    """Check if training script is ready to run"""
    print("\n" + "="*80)
    print("Training Script Readiness Check")
    print("="*80 + "\n")

    # 1. Check imports
    print("[1] Checking imports...")
    try:
        import torch
        from scripts.train import TrainingConfig, CondsarTrainer
        from models.enhanced_condsar import EnhancedDisasterControlNet
        print("[OK] All imports successful\n")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}\n")
        return False

    # 2. Check device
    print("[2] Checking device...")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    print(f"     Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\n")

    # 3. Create config
    print("[3] Creating config...")
    try:
        config = TrainingConfig(
            stage='a',
            source_dir='./data',
            batch_size=1,
            num_epochs=1
        )
        print(f"[OK] Config created")
        print(f"     Device: {config.device}")
        print(f"     Source dir: {config.source_dir}")
        print(f"     Output dir: {config.output_dir}\n")
    except Exception as e:
        print(f"[FAIL] Config creation failed: {e}\n")
        return False

    # 4. Check data
    print("[4] Checking data...")
    from models.training_utils import DisasterSARDataset
    try:
        dataset = DisasterSARDataset(
            dataset_dir=config.source_dir,
            image_size=config.image_size,
            return_mask=True,
            return_metadata=True
        )
        print(f"[OK] Dataset loaded")
        print(f"     Samples: {len(dataset)}\n")
    except Exception as e:
        print(f"[FAIL] Dataset loading failed: {e}\n")
        return False

    # 5. Create model
    print("[5] Creating model...")
    try:
        model = EnhancedDisasterControlNet(
            num_disaster_types=config.num_disaster_types,
            embedding_dim=config.embedding_dim,
            model_channels=config.model_channels
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created")
        print(f"     Parameters: {params:,}\n")
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}\n")
        return False

    print("="*80)
    print("[SUCCESS] All checks passed! Training script is ready!")
    print("="*80 + "\n")

    print("Quick start training:")
    print("  python scripts/train.py --stage a --num-epochs 1 --batch-size 1\n")

    return True

if __name__ == '__main__':
    success = test_training_ready()
    sys.exit(0 if success else 1)

