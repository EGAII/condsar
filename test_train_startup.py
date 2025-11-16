#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证训练脚本能否正确启动
仅检查数据加载部分，不实际训练
"""
import sys
from pathlib import Path
import argparse

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "models"))

def test_train_startup():
    """测试训练脚本启动"""
    print("\n" + "=" * 80)
    print("Testing Train Script Startup (Data Loading Only)")
    print("=" * 80 + "\n")

    from scripts.train import load_config_file, merge_config_with_args, TrainingConfig
    from models.training_utils import DisasterSARDataset

    # Step 1: 加载配置文件
    print("[1] Loading config_training.yaml...")
    config_dict = load_config_file('./config_training.yaml')
    if not config_dict:
        print("❌ Failed to load config")
        return False
    print(f"✅ Config loaded\n")

    # Step 2: 模拟命令行参数 (不指定--source-dir，让配置文件的值生效)
    print("[2] Creating command line arguments (no --source-dir specified)...")
    args = argparse.Namespace(
        config=None,
        stage='a',
        source_dir=None,  # 关键：不指定，让配置文件和默认值处理
        target_dir=None,
        batch_size=None,
        num_epochs=None,
        learning_rate=None,
        device=None,
        use_wandb=None,
        wandb_offline=False,
        output_dir=None,
        run_name=None
    )
    print(f"✅ Args created\n")

    # Step 3: 合并配置
    print("[3] Merging config with args...")
    merged_config = merge_config_with_args(config_dict, args)
    actual_source_dir = merged_config.get('source_dir', './data')
    print(f"✅ Config merged")
    print(f"   Final source_dir: {actual_source_dir}\n")

    # Step 4: 创建TrainingConfig
    print("[4] Creating TrainingConfig...")
    config = TrainingConfig(
        stage=merged_config.get('stage', 'a'),
        source_dir=merged_config.get('source_dir', './data'),
        target_dir=merged_config.get('target_dir', './data'),
        batch_size=merged_config.get('batch_size', 4),
        num_epochs=merged_config.get('num_epochs', 100),
        learning_rate=merged_config.get('learning_rate', 1e-4),
    )
    print(f"✅ TrainingConfig created")
    print(f"   source_dir: {config.source_dir}\n")

    # Step 5: 尝试加载数据集（关键测试）
    print("[5] Loading dataset (this is what was failing before)...")
    try:
        dataset = DisasterSARDataset(
            dataset_dir=config.source_dir,
            image_size=config.image_size,
            return_mask=True,
            return_metadata=True
        )
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}\n")

        print("[6] Loading first sample...")
        sample = dataset[0]
        print(f"✅ First sample loaded successfully")
        print(f"   Sample keys: {list(sample.keys())}\n")

        print("=" * 80)
        print("✅ STARTUP TEST PASSED - Training can proceed!")
        print("=" * 80 + "\n")
        return True

    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError: {e}\n")
        print("=" * 80)
        print("❌ STARTUP TEST FAILED!")
        print("=" * 80 + "\n")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ STARTUP TEST FAILED!")
        print("=" * 80 + "\n")
        return False

if __name__ == '__main__':
    success = test_train_startup()
    sys.exit(0 if success else 1)

