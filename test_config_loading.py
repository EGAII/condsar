#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整配置加载和数据验证测试
测试是否配置文件中的数据路径能正确加载数据
"""
import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "models"))

from models.training_utils import DisasterSARDataset
from scripts.train import load_config_file, merge_config_with_args, TrainingConfig
import argparse

def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 80)
    print("Testing Configuration Loading")
    print("=" * 80 + "\n")

    # Step 1: 加载配置文件
    print("[1] Loading config_training.yaml...")
    config_dict = load_config_file('./config_training.yaml')
    if config_dict:
        print(f"✅ Config loaded successfully")
        print(f"   source_dir from config: {config_dict.get('data', {}).get('source_dir', 'NOT FOUND')}")
    else:
        print(f"❌ Failed to load config")
        return False

    print()

    # Step 2: 检查数据路径
    print("[2] Checking source_dir setting...")
    source_dir = config_dict.get('data', {}).get('source_dir', './data')
    print(f"   source_dir: {source_dir}")

    # Step 3: 验证metadata.json是否存在
    print("[3] Verifying metadata.json exists...")
    metadata_path = Path(source_dir) / "metadata.json"
    if metadata_path.exists():
        print(f"✅ metadata.json found at: {metadata_path.absolute()}")
    else:
        print(f"❌ metadata.json NOT found at: {metadata_path.absolute()}")
        return False

    print()

    # Step 4: 模拟命令行参数
    print("[4] Simulating command line arguments...")
    args = argparse.Namespace(
        config=None,
        stage='a',
        source_dir=None,  # 让merge_config_with_args使用配置文件中的值
        target_dir=None,
        batch_size=4,
        num_epochs=100,
        learning_rate=1e-4,
        device=None,
        use_wandb=None,
        wandb_offline=False,
        output_dir=None,
        run_name=None
    )

    # 合并配置
    merged_config = merge_config_with_args(config_dict, args)
    print(f"✅ Config merged")
    print(f"   merged source_dir: {merged_config.get('source_dir', 'NOT FOUND')}")

    print()

    # Step 5: 创建TrainingConfig对象
    print("[5] Creating TrainingConfig object...")
    config = TrainingConfig(
        stage=merged_config.get('stage', 'a'),
        source_dir=merged_config.get('source_dir', './data'),
        target_dir=merged_config.get('target_dir', './data'),
        batch_size=merged_config.get('batch_size', 4),
        num_epochs=merged_config.get('num_epochs', 100),
        learning_rate=merged_config.get('learning_rate', 1e-4),
    )
    print(f"✅ TrainingConfig created")
    print(f"   config.source_dir: {config.source_dir}")

    print()

    # Step 6: 加载数据集
    print("[6] Loading dataset with config.source_dir...")
    try:
        dataset = DisasterSARDataset(
            dataset_dir=config.source_dir,
            image_size=config.image_size,
            return_mask=True,
            return_metadata=True
        )
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")

        # 尝试加载一个样本
        print("[7] Loading first sample...")
        sample = dataset[0]
        print(f"✅ Successfully loaded first sample")
        print(f"   Keys: {list(sample.keys())}")
        if 'rgb_image' in sample:
            print(f"   RGB image shape: {sample['rgb_image'].shape}")

        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80 + "\n")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80 + "\n")
        return False

if __name__ == '__main__':
    success = test_config_loading()
    sys.exit(0 if success else 1)

