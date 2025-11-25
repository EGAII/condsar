"""
Generate metadata.json for CONDSAR dataset

Scans pre/, post/, mask/ directories and creates metadata.json
with disaster type and severity information.

Usage:
    python generate_metadata_neds.py --data_dir ./condsar/data --output metadata.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DISASTER_TYPES = {
    'volcano': 0,
    'earthquake': 1,
    'wildfire': 2,
    'storm': 3,
    'flood': 4,
}

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']


def extract_disaster_type_from_filename(filename: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Extract disaster type and severity from filename

    Expected patterns:
    - volcano_0.5_image001.jpg -> type=0 (volcano), severity=0.5
    - earthquake_high_image002.jpg -> type=1 (earthquake), severity=0.8
    - wildfire_image003.jpg -> type=2 (wildfire), severity=0.5 (default)

    Args:
        filename: Image filename

    Returns:
        Tuple of (disaster_type_index, severity_value)
    """
    filename_lower = filename.lower()

    # Extract disaster type
    disaster_type = None
    for disaster_name, disaster_idx in DISASTER_TYPES.items():
        if disaster_name in filename_lower:
            disaster_type = disaster_idx
            break

    # Extract severity
    severity = None

    # Try to extract float value like "0.5" or "0.75"
    float_pattern = r'_(\d+\.\d+)_'
    match = re.search(float_pattern, filename)
    if match:
        try:
            severity = float(match.group(1))
            severity = max(0.0, min(1.0, severity))  # Clamp to [0, 1]
        except ValueError:
            pass

    # Try to extract severity keywords
    if severity is None:
        if 'low' in filename_lower or 'minor' in filename_lower:
            severity = 0.25
        elif 'medium' in filename_lower or 'moderate' in filename_lower:
            severity = 0.5
        elif 'high' in filename_lower or 'major' in filename_lower:
            severity = 0.75
        elif 'extreme' in filename_lower or 'severe' in filename_lower or 'destroyed' in filename_lower:
            severity = 1.0

    # Default severity
    if severity is None:
        severity = 0.5

    return disaster_type, severity


def extract_disaster_from_directory(file_path: Path, data_dir: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Extract disaster type from directory structure

    Expected structure:
    - data/pre/volcano/image001.jpg
    - data/pre/earthquake/region1/image002.jpg

    Args:
        file_path: Full file path
        data_dir: Base data directory

    Returns:
        Tuple of (disaster_type_index, severity_value)
    """
    try:
        relative_path = file_path.relative_to(data_dir)
        parts = relative_path.parts

        # Check directory names for disaster type
        for part in parts:
            part_lower = part.lower()
            for disaster_name, disaster_idx in DISASTER_TYPES.items():
                if disaster_name in part_lower:
                    return disaster_idx, 0.5  # Default severity

    except ValueError:
        pass

    return None, None


def find_matching_files(
    pre_dir: Path,
    post_dir: Path,
    mask_dir: Path,
    data_dir: Path
) -> Dict[str, Dict[str, any]]:
    """
    Find matching pre/post/mask triplets and generate metadata

    Args:
        pre_dir: Pre-disaster image directory
        post_dir: Post-disaster SAR directory
        mask_dir: Mask directory
        data_dir: Base data directory

    Returns:
        Dictionary of metadata entries
    """
    metadata = {}

    # Get all files in pre directory
    pre_files = []
    for ext in IMAGE_EXTENSIONS:
        pre_files.extend(list(pre_dir.rglob(f'*{ext}')))
        pre_files.extend(list(pre_dir.rglob(f'*{ext.upper()}')))

    logger.info(f"Found {len(pre_files)} files in pre directory")

    for pre_path in pre_files:
        # Get stem (filename without extension)
        stem = pre_path.stem

        # Find matching post file
        post_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = post_dir / f"{stem}{ext}"
            if candidate.exists():
                post_path = candidate
                break
            candidate = post_dir / f"{stem}{ext.upper()}"
            if candidate.exists():
                post_path = candidate
                break

        if post_path is None:
            logger.warning(f"No matching post file for {pre_path.name}, skipping")
            continue

        # Find matching mask file
        mask_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
            candidate = mask_dir / f"{stem}{ext.upper()}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            logger.warning(f"No matching mask file for {pre_path.name}, skipping")
            continue

        # Extract disaster type and severity
        # Priority: filename > directory structure > default
        disaster_type, severity = extract_disaster_type_from_filename(pre_path.name)

        if disaster_type is None:
            disaster_type, severity = extract_disaster_from_directory(pre_path, data_dir)

        if disaster_type is None:
            logger.warning(f"Could not determine disaster type for {pre_path.name}, using default (Flood)")
            disaster_type = 4  # Default to Flood
            severity = 0.5

        # Create relative paths
        try:
            pre_rel = pre_path.relative_to(data_dir)
            post_rel = post_path.relative_to(data_dir)
            mask_rel = mask_path.relative_to(data_dir)
        except ValueError as e:
            logger.error(f"Error creating relative paths: {e}")
            continue

        # Generate unique image ID
        image_id = stem
        counter = 1
        while image_id in metadata:
            image_id = f"{stem}_{counter}"
            counter += 1

        # Add to metadata
        metadata[image_id] = {
            "pre": str(pre_rel).replace('\\', '/'),  # Use forward slashes
            "post": str(post_rel).replace('\\', '/'),
            "mask": str(mask_rel).replace('\\', '/'),
            "disaster_type": int(disaster_type),
            "severity": float(severity),
        }

        logger.debug(f"Added {image_id}: type={disaster_type}, severity={severity}")

    return metadata


def generate_metadata(
    data_dir: str,
    output_file: str,
    pre_subdir: str = "pre",
    post_subdir: str = "post",
    mask_subdir: str = "mask",
):
    """
    Generate metadata.json for CONDSAR dataset

    Args:
        data_dir: Base data directory
        output_file: Output metadata.json path
        pre_subdir: Pre-disaster subdirectory name
        post_subdir: Post-disaster subdirectory name
        mask_subdir: Mask subdirectory name
    """
    data_dir = Path(data_dir)
    pre_dir = data_dir / pre_subdir
    post_dir = data_dir / post_subdir
    mask_dir = data_dir / mask_subdir

    logger.info("=" * 80)
    logger.info("CONDSAR Metadata Generation")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Pre directory: {pre_dir}")
    logger.info(f"Post directory: {post_dir}")
    logger.info(f"Mask directory: {mask_dir}")

    # Check directories exist
    if not pre_dir.exists():
        logger.error(f"Pre directory not found: {pre_dir}")
        return
    if not post_dir.exists():
        logger.error(f"Post directory not found: {post_dir}")
        return
    if not mask_dir.exists():
        logger.error(f"Mask directory not found: {mask_dir}")
        return

    # Find matching files
    logger.info("\nScanning for matching file triplets...")
    metadata = find_matching_files(pre_dir, post_dir, mask_dir, data_dir)

    logger.info(f"\nFound {len(metadata)} valid triplets")

    if len(metadata) == 0:
        logger.warning("No valid triplets found! Check your data structure.")
        return

    # Count disaster types
    type_counts = {}
    for entry in metadata.values():
        dtype = entry['disaster_type']
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    logger.info("\nDisaster type distribution:")
    for dtype, count in sorted(type_counts.items()):
        type_name = [k for k, v in DISASTER_TYPES.items() if v == dtype][0]
        logger.info(f"  {type_name} ({dtype}): {count} samples")

    # Save metadata
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ Metadata saved to: {output_path}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate metadata.json for CONDSAR dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base data directory containing pre/, post/, mask/ subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata.json",
        help="Output metadata.json file path (default: metadata.json in data_dir)"
    )
    parser.add_argument(
        "--pre_subdir",
        type=str,
        default="pre",
        help="Pre-disaster subdirectory name (default: pre)"
    )
    parser.add_argument(
        "--post_subdir",
        type=str,
        default="post",
        help="Post-disaster subdirectory name (default: post)"
    )
    parser.add_argument(
        "--mask_subdir",
        type=str,
        default="mask",
        help="Mask subdirectory name (default: mask)"
    )

    args = parser.parse_args()

    # Default output to data_dir/metadata.json if not absolute path
    output_file = args.output
    if not Path(output_file).is_absolute():
        output_file = str(Path(args.data_dir) / output_file)

    generate_metadata(
        data_dir=args.data_dir,
        output_file=output_file,
        pre_subdir=args.pre_subdir,
        post_subdir=args.post_subdir,
        mask_subdir=args.mask_subdir,
    )


if __name__ == "__main__":
    main()

