#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理脚本:
1. 将target文件夹改名为mask
2. 从文件名提取灾害类型
3. 过滤出5种灾害类型的数据
4. 删除其他灾害类型的数据

使用方法:
    python process_data.py [--dry-run]

参数:
    --dry-run: 仅显示操作，不真正执行删除
"""
import sys
from pathlib import Path

# 配置
DATA_DIR = Path('./data')
TARGET_DIR = DATA_DIR / 'target'
MASK_DIR = DATA_DIR / 'mask'

# 5种保留的灾害类型
DISASTER_TYPES = {
    'volcano': 'Volcano Eruption',
    'earthquake': 'Earthquake',
    'wildfire': 'Wildfire',
    'storm': 'Storm',
    'flood': 'Flood'
}

def extract_disaster_type(filename):
    """从文件名中提取灾害类型"""
    filename_lower = filename.lower()

    for disaster_key in DISASTER_TYPES.keys():
        if disaster_key in filename_lower:
            return disaster_key

    return None

def main():
    dry_run = '--dry-run' in sys.argv

    print("\n" + "=" * 90)
    print("Data Processing Script: Rename target to mask + Disaster Filtering")
    if dry_run:
        print("Mode: DRY RUN (Preview only, no deletion)")
    print("=" * 90 + "\n")

    # 第一步: 检查数据目录
    print("[1/4] 检查数据目录...")

    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} 不存在")
        sys.exit(1)

    print(f"OK: Data directory exists: {DATA_DIR.absolute()}\n")

    # 第二步: 检查当前目录结构
    print("[2/4] 检查当前目录结构...")
    pre_dir = DATA_DIR / 'pre'
    post_dir = DATA_DIR / 'post'

    print(f"  pre/: exists={pre_dir.exists()}")
    if pre_dir.exists():
        pre_files = list(pre_dir.glob('*'))
        print(f"       files={len(pre_files)}")

    print(f"  post/: exists={post_dir.exists()}")
    if post_dir.exists():
        post_files = list(post_dir.glob('*'))
        print(f"       files={len(post_files)}")

    print(f"  target/: exists={TARGET_DIR.exists()}")
    if TARGET_DIR.exists():
        target_files = list(TARGET_DIR.glob('*'))
        print(f"          files={len(target_files)}")

    print(f"  mask/: exists={MASK_DIR.exists()}")
    if MASK_DIR.exists():
        mask_files = list(MASK_DIR.glob('*'))
        print(f"        files={len(mask_files)}")

    # 第三步: 分析灾害类型分布
    print("\n[3/4] 分析文件和灾害类型分布...")

    disaster_distribution = {}
    unknown_files = []

    if pre_dir.exists():
        all_files = sorted(list(pre_dir.glob('*')))
        print(f"  扫描 {len(all_files)} 个文件...\n")

        for file_path in all_files:
            filename = file_path.name
            disaster = extract_disaster_type(filename)

            if disaster is None:
                unknown_files.append(filename)
            elif disaster in DISASTER_TYPES:
                if disaster not in disaster_distribution:
                    disaster_distribution[disaster] = 0
                disaster_distribution[disaster] += 1
            else:
                unknown_files.append(filename)

        print("灾害类型分布:")
        for disaster, count in sorted(disaster_distribution.items()):
            print(f"  {DISASTER_TYPES[disaster]:20s}: {count:5d} 个文件")

        if unknown_files:
            print(f"\n未知/其他类型文件数: {len(unknown_files)} 个")
            if len(unknown_files) <= 5:
                for f in unknown_files:
                    print(f"  - {f}")

    # 第四步: 执行改名和删除
    print("\n[4/4] 执行数据处理...\n")

    # 步骤A: 改名target→mask
    print("[步骤A] 将 target/ 改名为 mask/...")

    if TARGET_DIR.exists() and not MASK_DIR.exists():
        try:
            if not dry_run:
                TARGET_DIR.rename(MASK_DIR)
            print(f"  OK: target/ → mask/")
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    elif MASK_DIR.exists():
        print(f"  SKIP: mask/ 已存在")
    else:
        print(f"  SKIP: target/ 不存在")

    # 步骤B: 删除非5种灾害的文件
    print("\n[步骤B] 删除其他灾害类型的文件...")

    delete_count = 0
    delete_files = []

    for dirname in ['pre', 'post', 'mask']:
        dir_path = DATA_DIR / dirname
        if not dir_path.exists():
            continue

        files_in_dir = list(dir_path.glob('*'))

        for file_path in files_in_dir:
            filename = file_path.name
            disaster = extract_disaster_type(filename)

            if disaster not in DISASTER_TYPES or disaster is None:
                delete_files.append((dirname, filename))
                delete_count += 1

                if not dry_run:
                    try:
                        file_path.unlink()
                        print(f"  DELETE: {dirname}/{filename}")
                    except Exception as e:
                        print(f"  ERROR: {dirname}/{filename} - {e}")
                else:
                    print(f"  [DRY-RUN] DELETE: {dirname}/{filename}")

    print(f"\n  共 {'将删除' if dry_run else '已删除'}: {delete_count} 个文件")

    # 最终验证
    print("\n" + "=" * 90)
    print("最终结构验证")
    print("=" * 90)

    for dirname in ['pre', 'post', 'mask']:
        dir_path = DATA_DIR / dirname
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            print(f"\n{dirname}/: {len(files)} 个文件")

            # 统计灾害类型
            disaster_count = {}
            for f in files:
                disaster = extract_disaster_type(f.name)
                if disaster:
                    if disaster not in disaster_count:
                        disaster_count[disaster] = 0
                    disaster_count[disaster] += 1

            if disaster_count:
                for disaster in sorted(disaster_count.keys()):
                    print(f"  {DISASTER_TYPES.get(disaster, disaster):20s}: {disaster_count[disaster]:5d}")
            else:
                print("  (empty)")

    print("\n" + "=" * 90)
    if dry_run:
        print("DRY RUN Completed (No deletion executed)")
        print("Re-run without --dry-run parameter to execute actual processing")
    else:
        print("Data Processing Completed Successfully!")
    print("=" * 90 + "\n")

if __name__ == '__main__':
    main()

