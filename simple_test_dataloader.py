#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple DataLoader Test - Check TIF format handling
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "models"))

import torch
import tifffile
import numpy as np

print("\n" + "=" * 90)
print("DataLoader TIF Format Test")
print("=" * 90)

# Check data structure
data_dir = Path('./data')
target_dir = data_dir / 'target'

print("\n[1] Check directory structure...")
print("Data root: {}".format(data_dir.absolute()))
print("Target: {}".format(target_dir.absolute()))

if target_dir.exists():
    print("[OK] Target directory exists")

    # Find mask files
    mask_files = list(target_dir.glob('*_building_damage.tif'))
    print("Mask files found: {}".format(len(mask_files)))

    if len(mask_files) > 0:
        # Test first mask file
        sample_file = mask_files[0]
        print("\n[2] Analyzing sample mask file...")
        print("File: {}".format(sample_file.name))

        try:
            # Load with tifffile
            mask_data = tifffile.imread(str(sample_file))
            print("  Shape: {}".format(mask_data.shape))
            print("  Dtype: {}".format(mask_data.dtype))
            print("  Min: {}".format(mask_data.min()))
            print("  Max: {}".format(mask_data.max()))
            print("  Unique values: {}".format(np.unique(mask_data)))

            # Test normalization
            if mask_data.max() > 3:
                normalized = np.round(mask_data / 85).astype(np.uint8)
                normalized = np.clip(normalized, 0, 3)
                print("\n  After normalization:")
                print("    Min: {}".format(normalized.min()))
                print("    Max: {}".format(normalized.max()))
                print("    Unique: {}".format(np.unique(normalized)))
                print("  [OK] Mask can be normalized to [0-3]")

        except Exception as e:
            print("[ERROR] {}".format(e))
            import traceback
            traceback.print_exc()
else:
    print("[WARN] Target directory not found")

# Test DataLoader
print("\n[3] Test DisasterSARDataset...")

try:
    from models.training_utils import DisasterSARDataset

    # Try to load from target
    dataset = DisasterSARDataset(
        dataset_dir='./data/target',
        image_size=512,
        return_mask=True,
        return_metadata=False
    )

    print("[OK] Dataset created")
    print("Size: {}".format(len(dataset)))

    if len(dataset) > 0:
        print("\n[4] Load sample...")
        sample = dataset[0]
        print("Sample keys: {}".format(list(sample.keys())))

        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print("  {}: shape={}, dtype={}, range=[{:.3f}, {:.3f}]".format(
                    key, val.shape, val.dtype, val.min(), val.max()))

except Exception as e:
    print("[ERROR] {}".format(e))
    import traceback
    traceback.print_exc()

print("\n" + "=" * 90)
print("Test Complete")
print("=" * 90 + "\n")

