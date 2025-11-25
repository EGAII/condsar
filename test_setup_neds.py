#!/usr/bin/env python3
"""
Quick test script to validate CONDSAR NeDS installation and setup

Usage:
    python test_setup_neds.py
"""
import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported"""
    logger.info("Testing imports...")

    packages = {
        'torch': 'PyTorch',
        'diffusers': 'Diffusers',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tifffile': 'tifffile',
        'numpy': 'NumPy',
    }

    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            logger.info(f"  ✅ {name}")
        except ImportError:
            logger.error(f"  ❌ {name} not found")
            missing.append(name)

    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing.lower()))
        return False

    logger.info("All packages available ✅\n")
    return True


def test_cuda():
    """Test CUDA availability"""
    logger.info("Testing CUDA...")

    if torch.cuda.is_available():
        logger.info(f"  ✅ CUDA available")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        logger.warning("  ⚠️  CUDA not available, will use CPU")
        logger.warning("  Training will be VERY slow without GPU\n")

    return True


def test_model_creation():
    """Test that model can be created"""
    logger.info("Testing model creation...")

    try:
        from models.condsar_neds import CONDSARNeDS, SARVAEDecoder

        # Create small test model
        model = CONDSARNeDS(
            in_channels=4,
            conditioning_channels=1,
            block_out_channels=(64, 128, 256, 256),  # Smaller for testing
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            cross_attention_dim=None,
            num_disaster_types=5,
        )

        logger.info(f"  ✅ CONDSARNeDS model created")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create SAR decoder
        decoder = SARVAEDecoder(latent_channels=4, latent_size=64, output_channels=1)
        logger.info(f"  ✅ SARVAEDecoder created\n")

        return True

    except Exception as e:
        logger.error(f"  ❌ Model creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading"""
    logger.info("Testing dataset...")

    try:
        from models.training_utils import DisasterSARDataset

        # Check if test data exists
        test_data_dir = project_root / "condsar" / "data"
        if not test_data_dir.exists():
            logger.warning(f"  ⚠️  Data directory not found: {test_data_dir}")
            logger.warning("  Create data directory and run generate_metadata_neds.py")
            return True

        metadata_file = test_data_dir / "metadata.json"
        if not metadata_file.exists():
            logger.warning(f"  ⚠️  metadata.json not found")
            logger.warning("  Run: python generate_metadata_neds.py --data_dir ./condsar/data")
            return True

        # Try to load dataset
        dataset = DisasterSARDataset(
            dataset_dir=str(test_data_dir),
            image_size=512,
            return_mask=True,
            return_metadata=True,
        )

        logger.info(f"  ✅ Dataset loaded: {len(dataset)} samples")

        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"  Sample keys: {list(sample.keys())}")
            logger.info(f"  RGB shape: {sample['rgb_image'].shape}")
            logger.info(f"  SAR shape: {sample['sar_image'].shape}")
            logger.info(f"  Mask shape: {sample['building_mask'].shape}")
            logger.info(f"  Disaster type: {sample['disaster_type'].item()}")
            logger.info(f"  Severity: {sample['disaster_severity'].item():.2f}\n")

        return True

    except Exception as e:
        logger.error(f"  ❌ Dataset test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass through the model"""
    logger.info("Testing forward pass...")

    try:
        from models.condsar_neds import CONDSARNeDS

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create minimal model
        model = CONDSARNeDS(
            in_channels=4,
            conditioning_channels=1,
            block_out_channels=(64, 128, 256, 256),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            cross_attention_dim=None,
            num_disaster_types=5,
        ).to(device)

        # Create dummy inputs
        batch_size = 2
        sample = torch.randn(batch_size, 4, 16, 16).to(device)
        pre_latents = torch.randn(batch_size, 4, 16, 16).to(device)
        mask = torch.randint(0, 4, (batch_size, 1, 16, 16)).float().to(device)
        disaster_type = torch.randint(0, 5, (batch_size,)).to(device)
        disaster_severity = torch.rand(batch_size).to(device)
        timestep = torch.tensor([100]).to(device)
        encoder_hidden_states = torch.randn(batch_size, 1, 64).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pre_latents=pre_latents,
                mask=mask,
                disaster_type=disaster_type,
                disaster_severity=disaster_severity,
                return_dict=True,
            )

        logger.info(f"  ✅ Forward pass successful")
        logger.info(f"  Output type: {type(output)}")
        logger.info(f"  Down samples: {len(output.down_block_res_samples)} blocks")
        logger.info(f"  Mid sample shape: {output.mid_block_res_sample.shape}\n")

        return True

    except Exception as e:
        logger.error(f"  ❌ Forward pass failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("CONDSAR NeDS Setup Test")
    logger.info("=" * 80 + "\n")

    tests = [
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("Model Creation", test_model_creation),
        ("Dataset Loading", test_dataset),
        ("Forward Pass", test_forward_pass),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}\n")
            results.append((test_name, False))

    # Summary
    logger.info("=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    logger.info(f"\n{passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! You're ready to train CONDSAR NeDS.")
        logger.info("\nNext steps:")
        logger.info("  1. Organize your data in condsar/data/pre, post, mask")
        logger.info("  2. Run: python generate_metadata_neds.py --data_dir ./condsar/data")
        logger.info("  3. Run: python train_neds.py --config config_neds.yaml --stage a")
    else:
        logger.error("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

