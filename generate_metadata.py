#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动生成 metadata.json 脚本
从数据文件夹结构和文件名生成元数据

使用方法:
    python generate_metadata.py [data_dir] [--output-file metadata.json]
"""
import sys
import json
from pathlib import Path
from typing import Dict, Optional

# 灾害类型映射 (移除了Storm)
DISASTER_KEYWORDS = {
    'volcano': 0,
    'earthquake': 1,
    'wildfire': 2,
    'flood': 3
}

DISASTER_TYPES = {
    0: "Volcano",
    1: "Earthquake",
    2: "Wildfire",
    3: "Flood"
}


def extract_disaster_type(filename: str) -> int:
    """从文件名中提取灾害类型 (0-3)"""
    filename_lower = filename.lower()

    for keyword, disaster_id in DISASTER_KEYWORDS.items():
        if keyword in filename_lower:
            return disaster_id

    # 默认返回0
    return 0


def extract_image_id(filename: str) -> str:
    """从文件名中提取图像ID (去掉扩展名和后缀)"""
    # 处理格式: congo-volcano_00000000_pre_disaster.jpg
    # 返回: congo-volcano_00000000

    # 去掉扩展名
    stem = Path(filename).stem

    # 去掉 _pre_disaster 或 _post_disaster 或 _building_damage 后缀
    for suffix in ['_pre_disaster', '_post_disaster', '_building_damage']:
        if suffix in stem:
            stem = stem.replace(suffix, '')

    return stem


def find_matching_files(data_dir: Path, image_id: str) -> Dict[str, Optional[str]]:
    """为给定的image_id查找对应的pre, post, mask文件"""
    pre_dir = data_dir / "pre"
    post_dir = data_dir / "post"
    mask_dir = data_dir / "mask"

    result = {
        'pre': None,
        'post': None,
        'mask': None
    }

    # 查找pre文件
    if pre_dir.exists():
        for f in pre_dir.glob('*'):
            if image_id in f.stem:
                result['pre'] = str(f.relative_to(data_dir))
                break

    # 查找post文件
    if post_dir.exists():
        for f in post_dir.glob('*'):
            if image_id in f.stem:
                result['post'] = str(f.relative_to(data_dir))
                break

    # 查找mask文件
    if mask_dir.exists():
        for f in mask_dir.glob('*'):
            if image_id in f.stem:
                result['mask'] = str(f.relative_to(data_dir))
                break

    return result


def generate_metadata(data_dir: str = './data', output_file: str = 'metadata.json') -> Dict:
    """
    生成metadata.json

    Args:
        data_dir: 数据目录路径
        output_file: 输出文件名

    Returns:
        生成的metadata字典
    """
    data_dir = Path(data_dir)

    print("\n" + "=" * 90)
    print("Generate metadata.json from data structure")
    print("=" * 90 + "\n")

    # 检查目录
    print("[1/4] Checking data directory...")
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return {}

    pre_dir = data_dir / "pre"
    if not pre_dir.exists():
        print(f"ERROR: pre/ directory not found in {data_dir}")
        return {}

    print(f"OK: Data directory exists: {data_dir.absolute()}\n")

    # 扫描pre目录获取所有图像
    print("[2/4] Scanning pre/ directory for images...")
    pre_files = sorted([f for f in pre_dir.glob('*') if f.is_file()])
    print(f"Found {len(pre_files)} images in pre/\n")

    # 提取所有唯一的image_id
    print("[3/4] Extracting image IDs...")
    image_ids = set()
    for f in pre_files:
        image_id = extract_image_id(f.name)
        image_ids.add(image_id)

    image_ids = sorted(list(image_ids))
    print(f"Unique image IDs: {len(image_ids)}\n")

    # 生成metadata
    print("[4/4] Generating metadata...")
    metadata = {}
    missing_count = 0

    for i, image_id in enumerate(image_ids):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(image_ids)}...")

        # 查找对应的文件
        files = find_matching_files(data_dir, image_id)

        # 检查必需的文件
        if not files['pre']:
            print(f"WARNING: Missing pre file for {image_id}")
            missing_count += 1
            continue

        # 从文件名提取灾害类型
        disaster_type = extract_disaster_type(image_id)

        # 创建metadata条目
        metadata[image_id] = {
            'pre': files['pre'],
            'post': files['post'],
            'mask': files['mask'],
            'disaster_type': disaster_type,
            'severity': 0.5  # 默认强度
        }

    print(f"  Processed {len(image_ids)}/{len(image_ids)} total\n")

    if missing_count > 0:
        print(f"WARNING: {missing_count} images with missing pre files were skipped\n")

    # 统计灾害类型
    print("Disaster type distribution:")
    disaster_count = {}
    for entry in metadata.values():
        dtype = entry['disaster_type']
        if dtype not in disaster_count:
            disaster_count[dtype] = 0
        disaster_count[dtype] += 1

    for dtype_id in sorted(disaster_count.keys()):
        dtype_name = DISASTER_TYPES.get(dtype_id, "Unknown")
        count = disaster_count[dtype_id]
        print(f"  {dtype_name:20s}: {count:5d}")

    print(f"\nTotal entries: {len(metadata)}\n")

    # 保存到文件
    output_path = data_dir / output_file
    print(f"Saving metadata to: {output_path.absolute()}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("OK: Metadata saved successfully\n")

    print("=" * 90)
    print("Metadata generation completed!")
    print("=" * 90 + "\n")

    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate metadata.json from data structure')
    parser.add_argument('data_dir', nargs='?', default='./data', help='Data directory path')
    parser.add_argument('--output-file', default='metadata.json', help='Output metadata filename')

    args = parser.parse_args()

    metadata = generate_metadata(args.data_dir, args.output_file)

    if metadata:
        print("SUCCESS: metadata.json generated!")
        return 0
    else:
        print("FAILED: Could not generate metadata.json")
        return 1


if __name__ == '__main__':
    sys.exit(main())

