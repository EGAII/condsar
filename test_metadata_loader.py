#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的 DisasterSARDataset
验证从 metadata.json 加载数据
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

import torch
from models.training_utils import DisasterSARDataset

print("\n" + "=" * 90)
print("Test: DisasterSARDataset with metadata.json")
print("=" * 90 + "\n")

# Test 1: Create dataset
print("[TEST 1] Create dataset...")
try:
    dataset = DisasterSARDataset(
        dataset_dir='./data',
        image_size=512,
        return_mask=True,
        return_metadata=True
    )
    print("OK: Dataset created successfully")
    print(f"    Dataset size: {len(dataset)}")
    print(f"    Disaster types: {list(dataset.DISASTER_TYPES.values())}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load sample
print("\n[TEST 2] Load and verify sample...")
if len(dataset) > 0:
    try:
        sample = dataset[0]
        print("OK: Sample loaded")
        print(f"    Keys: {list(sample.keys())}")
        print(f"    Image name: {sample['image_id']}")

        # Check shapes
        print(f"    RGB shape: {sample['rgb_image'].shape}")
        print(f"    SAR shape: {sample['sar_image'].shape}")
        print(f"    Mask shape: {sample['building_mask'].shape}")

        # Check disaster info
        disaster_type = sample['disaster_type'].item()
        severity = sample['disaster_severity'].item()
        print(f"    Disaster type: {disaster_type} ({dataset.DISASTER_TYPES.get(disaster_type, 'Unknown')})")
        print(f"    Severity: {severity:.2f}")

        # Check image values
        rgb_min, rgb_max = sample['rgb_image'].min().item(), sample['rgb_image'].max().item()
        print(f"    RGB value range: [{rgb_min:.3f}, {rgb_max:.3f}]")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 3: Check disaster type distribution
print("\n[TEST 3] Check disaster type distribution...")
disaster_dist = {}
try:
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        disaster_type = sample['disaster_type'].item()
        if disaster_type not in disaster_dist:
            disaster_dist[disaster_type] = 0
        disaster_dist[disaster_type] += 1
except Exception as e:
    print(f"ERROR: {e}")

print("Disaster types found in first 100 samples:")
for disaster_id in sorted(disaster_dist.keys()):
    count = disaster_dist[disaster_id]
    name = dataset.DISASTER_TYPES.get(disaster_id, "Unknown")
    print(f"  {disaster_id}: {name:20s} - {count:3d} samples")

# Test 4: Check batch loading
print("\n[TEST 4] Test batch loading...")
try:
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print("OK: Batch loaded")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 90)
print("All tests completed successfully!")
print("=" * 90 + "\n")

