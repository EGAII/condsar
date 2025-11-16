#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据加载是否正常工作
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

from models.training_utils import DisasterSARDataset

def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("Testing Data Loading")
    print("=" * 80 + "\n")

    # 测试 source_dir = './data'
    print("[1] Testing dataset loading from './data'...")
    try:
        dataset = DisasterSARDataset(
            dataset_dir='./data',
            image_size=512,
            return_mask=True,
            return_metadata=True
        )
        print(f"✅ Successfully loaded dataset with {len(dataset)} samples\n")

        # 尝试加载第一个样本
        print("[2] Testing first sample loading...")
        sample = dataset[0]
        print(f"✅ Successfully loaded first sample")
        print(f"   Keys: {sample.keys()}")
        if 'rgb_image' in sample:
            print(f"   RGB image shape: {sample['rgb_image'].shape}")
        if 'sar_image' in sample:
            print(f"   SAR image shape: {sample['sar_image'].shape}")
        if 'building_mask' in sample:
            print(f"   Building mask shape: {sample['building_mask'].shape}")
        print()

        print("=" * 80)
        print("✅ Data loading test PASSED!")
        print("=" * 80 + "\n")
        return True

    except Exception as e:
        print(f"❌ Error during data loading: {e}\n")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ Data loading test FAILED!")
        print("=" * 80 + "\n")
        return False

if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)

