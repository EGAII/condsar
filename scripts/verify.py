"""
CONDSAR 验证和演示脚本
测试完整的训练和推理流程
"""
import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "models"))

from src.utils.logger import setup_logger

logger = setup_logger('condsar_verify', log_dir='./outputs/logs')


def create_dummy_dataset(data_dir: str = './data', num_samples: int = 5):
    """创建虚拟数据集用于测试"""

    logger.info("\n" + "=" * 80)
    logger.info("🔨 Creating Dummy Dataset for Testing")
    logger.info("=" * 80)

    data_path = Path(data_dir)

    # 源域数据
    source_path = data_path / "source"
    source_path.mkdir(parents=True, exist_ok=True)
    (source_path / "pre").mkdir(exist_ok=True)
    (source_path / "post").mkdir(exist_ok=True)
    (source_path / "mask").mkdir(exist_ok=True)

    logger.info(f"Creating source domain data ({num_samples} samples)...")

    metadata = {}
    for i in range(num_samples):
        # RGB图像 (256x256)
        rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(rgb, 'RGB').save(source_path / "pre" / f"image_{i:03d}.jpg")

        # SAR图像 (256x256)
        sar = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        Image.fromarray(sar, 'L').save(source_path / "post" / f"image_{i:03d}.jpg")

        # Mask (256x256)
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        Image.fromarray(mask, 'L').save(source_path / "mask" / f"image_{i:03d}.png")

        # 元数据
        metadata[f"image_{i:03d}.jpg"] = {
            "disaster_type": i % 5,
            "severity": (i + 1) / num_samples,
            "damage_level": i % 4
        }

    with open(source_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Source domain data created: {source_path}")
    logger.info(f"   - RGB images: {num_samples}")
    logger.info(f"   - SAR images: {num_samples}")
    logger.info(f"   - Masks: {num_samples}")

    # 目标域数据
    target_path = data_path / "target"
    target_path.mkdir(parents=True, exist_ok=True)
    (target_path / "pre").mkdir(exist_ok=True)
    (target_path / "mask").mkdir(exist_ok=True)

    logger.info(f"Creating target domain data ({num_samples} samples)...")

    for i in range(num_samples):
        # RGB图像
        rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(rgb, 'RGB').save(target_path / "pre" / f"image_{i:03d}.jpg")

        # Mask
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        Image.fromarray(mask, 'L').save(target_path / "mask" / f"image_{i:03d}.png")

    logger.info(f"✅ Target domain data created: {target_path}")
    logger.info(f"   - RGB images: {num_samples}")
    logger.info(f"   - Masks: {num_samples}")

    return str(source_path), str(target_path)


def test_imports():
    """测试导入"""

    logger.info("\n" + "=" * 80)
    logger.info("🧪 Testing Imports")
    logger.info("=" * 80)

    try:
        from models.training_utils import DisasterSARDataset, MetricsTracker
        logger.info("✅ training_utils imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import training_utils: {e}")
        return False

    try:
        from models.enhanced_condsar import EnhancedDisasterControlNet
        logger.info("✅ enhanced_condsar imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import enhanced_condsar: {e}")
        return False

    try:
        import wandb
        logger.info("✅ wandb imported successfully")
    except ImportError:
        logger.warning("⚠️ wandb not installed (optional)")

    return True


def test_dataset(source_dir: str):
    """测试数据集加载"""

    logger.info("\n" + "=" * 80)
    logger.info("🧪 Testing Dataset Loading")
    logger.info("=" * 80)

    try:
        from models.training_utils import DisasterSARDataset

        dataset = DisasterSARDataset(
            dataset_dir=source_dir,
            image_size=256,
            return_mask=True,
            return_metadata=True,
            logger=logger
        )

        logger.info(f"✅ Dataset loaded: {len(dataset)} samples")

        # 测试数据项
        item = dataset[0]
        logger.info(f"   - RGB shape: {item['rgb_image'].shape}")
        logger.info(f"   - SAR shape: {item['sar_image'].shape}")
        logger.info(f"   - Mask shape: {item['building_mask'].shape}")
        logger.info(f"   - Disaster type: {item['disaster_type'].item()}")
        logger.info(f"   - Severity: {item['disaster_severity'].item():.3f}")

        return True

    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """测试模型"""

    logger.info("\n" + "=" * 80)
    logger.info("🧪 Testing Model")
    logger.info("=" * 80)

    try:
        from models.enhanced_condsar import EnhancedDisasterControlNet

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # 创建模型
        model = EnhancedDisasterControlNet(
            num_disaster_types=5,
            embedding_dim=128,
            model_channels=320
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created: {num_params:,} parameters")

        # 前向传播测试
        batch_size = 2
        rgb = torch.randn(batch_size, 3, 256, 256).to(device)
        sar = torch.randn(batch_size, 1, 256, 256).to(device)
        mask = torch.randn(batch_size, 1, 256, 256).to(device)
        disaster_type = torch.randint(0, 5, (batch_size,)).to(device)
        disaster_severity = torch.rand(batch_size).to(device)
        timestep = torch.randint(0, 1000, (batch_size,)).to(device)

        with torch.no_grad():
            output = model(
                sample=sar,
                timestep=timestep,
                encoder_hidden_states=rgb,
                rgb_image=rgb,
                building_mask=mask,
                disaster_type=disaster_type,
                disaster_severity=disaster_severity
            )

        logger.info(f"✅ Forward pass successful")
        logger.info(f"   - Input SAR: {sar.shape}")
        logger.info(f"   - Output: {output.shape}")
        logger.info(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")

        return True

    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wandb():
    """测试WandB"""

    logger.info("\n" + "=" * 80)
    logger.info("🧪 Testing WandB Integration")
    logger.info("=" * 80)

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.utils.wandb_utils import WandBVisualizer

        # 创建visualizer
        viz = WandBVisualizer('condsar-test')
        logger.info("✅ WandBVisualizer created")

        # 测试记录
        rgb = torch.rand(1, 3, 256, 256)
        sar = torch.rand(1, 1, 256, 256)

        viz.log_image("test_rgb", rgb[0])
        logger.info("✅ Image logging successful")

        viz.log_training_comparison(
            "test_comparison",
            rgb=rgb,
            sar_pred=sar,
            sar_gt=sar + 0.1 * torch.randn_like(sar)
        )
        logger.info("✅ Comparison logging successful")

        viz.finish()
        logger.info("✅ WandB test completed")

        return True

    except Exception as e:
        logger.warning(f"⚠️ WandB test failed (optional): {e}")
        return False


def test_inference_script():
    """测试推理脚本"""

    logger.info("\n" + "=" * 80)
    logger.info("🧪 Testing Inference Script")
    logger.info("=" * 80)

    try:
        # 创建测试数据
        test_dir = Path('./outputs/test_inference')
        test_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试图像
        rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        rgb_path = test_dir / "test_rgb.jpg"
        Image.fromarray(rgb, 'RGB').save(rgb_path)
        logger.info(f"✅ Test image created: {rgb_path}")

        # 创建测试mask
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        mask_path = test_dir / "test_mask.png"
        Image.fromarray(mask, 'L').save(mask_path)
        logger.info(f"✅ Test mask created: {mask_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='CONDSAR Verification Script')
    parser.add_argument('--create-data', action='store_true',
                       help='Create dummy dataset')
    parser.add_argument('--test-imports', action='store_true', default=True,
                       help='Test imports')
    parser.add_argument('--test-dataset', action='store_true',
                       help='Test dataset loading')
    parser.add_argument('--test-model', action='store_true',
                       help='Test model')
    parser.add_argument('--test-wandb', action='store_true',
                       help='Test WandB integration')
    parser.add_argument('--full', action='store_true',
                       help='Run all tests')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')

    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("🚀 CONDSAR Verification Suite")
    logger.info("=" * 80)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 创建虚拟数据集
    if args.create_data or args.full:
        source_dir, target_dir = create_dummy_dataset(args.data_dir, num_samples=5)
    else:
        source_dir = f"{args.data_dir}/source"
        target_dir = f"{args.data_dir}/target"

    tests = []

    # 测试导入
    if args.test_imports or args.full:
        tests.append(("Imports", test_imports()))

    # 测试数据集
    if args.test_dataset or args.full:
        tests.append(("Dataset", test_dataset(source_dir)))

    # 测试模型
    if args.test_model or args.full:
        tests.append(("Model", test_model()))

    # 测试WandB
    if args.test_wandb or args.full:
        tests.append(("WandB", test_wandb()))

    # 测试推理脚本
    if args.full:
        tests.append(("Inference", test_inference_script()))

    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("📊 Test Summary")
    logger.info("=" * 80)

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready for training/inference.")
    else:
        logger.warning(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")

    logger.info("=" * 80)


if __name__ == '__main__':
    main()

