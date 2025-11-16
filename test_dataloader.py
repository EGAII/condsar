#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataLoader Test Script - Test loading actual disaster data
Test: post, pre, target folders with TIF format support
With detailed logger debug output
"""
import sys
import os
import logging
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "models"))

import torch
import tifffile
import numpy as np
from models.training_utils import DisasterSARDataset
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(
    name='dataloader_test',
    log_dir='./outputs/logs',
    level=logging.DEBUG
)

print("\n" + "=" * 90)
print("DataLoader Test - Real Data Structure Analysis")
print("=" * 90 + "\n")

logger.info("=" * 90)
logger.info("DataLoader Test - Real Data Structure Analysis")
logger.info("=" * 90)

# Test actual data location
data_root = Path('./data')
target_dir = data_root / 'target'

logger.info("\n[TEST 1] Check data directory structure...")
logger.info("Data root: {}".format(data_root.absolute()))
logger.info("Target directory: {}".format(target_dir.absolute()))

if target_dir.exists():
    logger.info("[OK] Target directory exists")

    # List subdirectories
    subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
    logger.info("Subdirectories found: {}".format(len(subdirs)))
    for subdir in subdirs:
        logger.info("  - {}".format(subdir.name))

    # Check for mask files directly in target
    mask_files = list(target_dir.glob('*_building_damage.tif'))
    logger.info("\nMask files directly in target/: {}".format(len(mask_files)))
    if len(mask_files) > 0:
        logger.info("  Sample files:")
        for f in mask_files[:5]:
            logger.info("    * {}".format(f.name))
        if len(mask_files) > 5:
            logger.info("    ... and {} more".format(len(mask_files) - 5))
else:
    logger.warning("[WARN] Target directory does not exist")

# Test actual directory structure expected by DataLoader
logger.info("\n[TEST 2] Check expected DataLoader structure...")

expected_structure = {
    'pre': 'RGB images (灾前)',
    'post': 'SAR images (灾后)',
    'mask': 'Building masks (建筑掩码)'
}

for dirname, description in expected_structure.items():
    dirpath = target_dir / dirname
    if dirpath.exists():
        files = list(dirpath.glob('*'))
        logger.info("[OK] {}: exists ({} files)".format(dirname, len(files)))
        if len(files) > 0 and len(files) <= 3:
            for f in files:
                logger.info("      * {}".format(f.name))
    else:
        logger.warning("[WARN] {}: NOT found".format(dirname))

# Test actual data file
logger.info("\n[TEST 3] Analyzing actual mask file...")

sample_mask_path = list(target_dir.glob('*_building_damage.tif'))
if len(sample_mask_path) > 0:
    sample_mask_path = sample_mask_path[0]
    logger.info("Sample file: {}".format(sample_mask_path.name))

    try:
        # Load with tifffile
        mask_data = tifffile.imread(str(sample_mask_path))
        logger.info("  Shape: {}".format(mask_data.shape))
        logger.info("  Dtype: {}".format(mask_data.dtype))
        logger.info("  Min value: {}".format(mask_data.min()))
        logger.info("  Max value: {}".format(mask_data.max()))
        logger.info("  Unique values: {}".format(np.unique(mask_data).tolist()))

        # Check if normalization is needed
        if mask_data.max() > 3:
            normalized = np.round(mask_data / 85).astype(np.uint8)
            normalized = np.clip(normalized, 0, 3)
            logger.info("  After normalization: min={}, max={}, unique={}".format(
                normalized.min(), normalized.max(), np.unique(normalized).tolist()))
            logger.info("[OK] Mask can be normalized to [0-3] range")
        else:
            logger.info("[OK] Mask already in [0-3] range")

    except Exception as e:
        logger.error("[ERROR] Failed to load mask: {}".format(e))

# Test DataLoader creation
logger.info("\n[TEST 4] Test DataLoader creation...")

# Create a structured test directory
test_structure_dir = Path('./data_test_structure')
test_structure_dir.mkdir(exist_ok=True)

logger.info("Creating test structure in: {}".format(test_structure_dir))

# Create required directories
for dirname in ['pre', 'post', 'mask']:
    (test_structure_dir / dirname).mkdir(exist_ok=True)
    logger.info("  Created: {}".format(dirname))

# Copy sample files to test structure
logger.info("\nCopying sample files for testing...")

mask_files = list(target_dir.glob('*_building_damage.tif'))[:3]
for mask_file in mask_files:
    new_name = mask_file.name.replace('_building_damage.tif', '.tif')
    dest_path = test_structure_dir / 'mask' / new_name

    # Create dummy RGB and SAR files
    logger.debug("  Preparing sample: {}".format(new_name))

    # Copy mask
    try:
        mask_data = tifffile.imread(str(mask_file))
        tifffile.imwrite(str(dest_path), mask_data)
        logger.info("  [OK] Copied mask: {}".format(new_name))
    except Exception as e:
        logger.error("    [ERROR] Failed to copy mask: {}".format(e))

    # Create dummy RGB (RGB image)
    try:
        rgb_dummy = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        rgb_path = test_structure_dir / 'pre' / new_name
        tifffile.imwrite(str(rgb_path), rgb_dummy)
        logger.debug("    Created dummy RGB")
    except Exception as e:
        logger.error("    [ERROR] Failed to create RGB: {}".format(e))

    # Create dummy SAR (grayscale image)
    try:
        sar_dummy = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        sar_path = test_structure_dir / 'post' / new_name
        tifffile.imwrite(str(sar_path), sar_dummy)
        logger.debug("    Created dummy SAR")
    except Exception as e:
        logger.error("    [ERROR] Failed to create SAR: {}".format(e))

# Now test DataLoader
logger.info("\n[TEST 5] Test DataLoader with test structure...")

try:
    dataset = DisasterSARDataset(
        dataset_dir=str(test_structure_dir),
        image_size=512,
        return_mask=True,
        return_metadata=False,
        logger=logger
    )

    logger.info("[OK] DataLoader created successfully")
    logger.info("Dataset size: {}".format(len(dataset)))

    if len(dataset) > 0:
        logger.info("\nLoading sample from dataset...")
        sample = dataset[0]

        logger.info("Sample keys: {}".format(list(sample.keys())))

        if 'rgb_image' in sample:
            rgb = sample['rgb_image']
            logger.info("RGB: shape={}, dtype={}, range=[{:.3f}, {:.3f}]".format(
                rgb.shape, rgb.dtype, rgb.min(), rgb.max()))

        if 'sar_image' in sample:
            sar = sample['sar_image']
            logger.info("SAR: shape={}, dtype={}, range=[{:.3f}, {:.3f}]".format(
                sar.shape, sar.dtype, sar.min(), sar.max()))

        if 'building_mask' in sample:
            mask = sample['building_mask']
            logger.info("Mask: shape={}, dtype={}, range=[{:.3f}, {:.3f}]".format(
                mask.shape, mask.dtype, mask.min(), mask.max()))

            unique_vals = torch.unique(mask).tolist()
            logger.info("Mask unique values: {}".format(unique_vals))

            if mask.max() <= 3:
                logger.info("[OK] Mask successfully normalized to [0-3]")
            else:
                logger.warning("[WARN] Mask not in [0-3] range: max={}".format(mask.max()))

        # Test batch loading
        logger.info("\n[TEST 6] Test batch loading...")
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(dataloader))
        logger.info("[OK] Batch loaded successfully")
        logger.info("Batch keys: {}".format(list(batch.keys())))

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info("  {}: {}".format(key, value.shape))

except Exception as e:
    logger.error("[ERROR] DataLoader test failed: {}".format(e))
    import traceback
    logger.error(traceback.format_exc())

# Cleanup
logger.info("\n[CLEANUP] Removing test structure...")
import shutil
try:
    shutil.rmtree(test_structure_dir)
    logger.info("[OK] Test structure removed")
except Exception as e:
    logger.warning("[WARN] Failed to cleanup: {}".format(e))

# Summary
logger.info("\n" + "=" * 90)
logger.info("Test Summary")
logger.info("=" * 90)

logger.info("\nYour data structure in data/target/:")
logger.info("  - Contains mask files: *_building_damage.tif")
logger.info("  - Format: TIF (supported)")
logger.info("  - Values: [0-255] (auto-normalized to [0-3])")

logger.info("\nTo use DataLoader, you need:")
logger.info("  data/target/")
logger.info("    pre/      (RGB images)")
logger.info("    post/     (SAR images)")
logger.info("    mask/     (Building masks)")

logger.info("\nLog saved to: ./outputs/logs/\n")



