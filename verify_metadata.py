#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 metadata.json 数据完整性和正确性
并生成灾害类型数据分布的可视化图表
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Disaster type mapping
DISASTER_TYPES = {
    0: "Volcano",
    1: "Earthquake",
    2: "Wildfire",
    3: "Flood"
}

def verify_metadata(metadata_file: str = './data/metadata.json') -> Dict:
    """Verify the completeness and correctness of metadata.json"""

    print("\n" + "=" * 100)
    print("Verifying metadata.json")
    print("=" * 100 + "\n")

    metadata_path = Path(metadata_file)

    # Check file existence
    print("[1/5] Checking file...")
    if not metadata_path.exists():
        print(f"ERROR: {metadata_file} not found")
        return {}

    print(f"OK: File exists ({metadata_path.absolute()})")
    print(f"    File size: {metadata_path.stat().st_size / 1024:.2f} KB\n")

    # Load metadata
    print("[2/5] Loading metadata...")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"OK: Loaded successfully")
        print(f"    Total entries: {len(metadata)}\n")
    except Exception as e:
        print(f"ERROR: Load failed - {e}")
        return {}

    # Verify data structure
    print("[3/5] Verifying data structure...")
    required_fields = {'pre', 'post', 'mask', 'disaster_type', 'severity'}
    missing_entries = []
    incomplete_entries = []

    for image_id, entry in metadata.items():
        if not isinstance(entry, dict):
            incomplete_entries.append((image_id, "Value is not a dict"))
            continue

        missing = required_fields - set(entry.keys())
        if missing:
            incomplete_entries.append((image_id, f"Missing fields: {missing}"))

    if missing_entries or incomplete_entries:
        print(f"WARNING: Found incomplete entries")
        if missing_entries:
            print(f"  Missing entries: {len(missing_entries)}")
        if incomplete_entries:
            print(f"  Incomplete entries: {len(incomplete_entries)}")
            for img_id, reason in incomplete_entries[:5]:
                print(f"    - {img_id}: {reason}")
            if len(incomplete_entries) > 5:
                print(f"    ... and {len(incomplete_entries) - 5} more")
    else:
        print("OK: All entries have complete structure\n")

    # Verify file paths
    print("[4/5] Verifying file paths...")
    data_dir = metadata_path.parent
    missing_files = {
        'pre': [],
        'post': [],
        'mask': []
    }

    for image_id, entry in metadata.items():
        for file_type in ['pre', 'post', 'mask']:
            file_path = data_dir / entry[file_type]
            if not file_path.exists():
                missing_files[file_type].append((image_id, entry[file_type]))

    all_ok = True
    for file_type, missing_list in missing_files.items():
        if missing_list:
            print(f"WARNING: {file_type} directory missing {len(missing_list)} files")
            all_ok = False
            for img_id, path in missing_list[:3]:
                print(f"  - {img_id}: {path}")
            if len(missing_list) > 3:
                print(f"  ... and {len(missing_list) - 3} more")
        else:
            print(f"OK: All files in {file_type} directory exist")

    if all_ok:
        print()

    # Collect disaster type statistics
    print("[5/5] Collecting disaster type statistics...")
    disaster_stats = {}
    invalid_disasters = []

    for image_id, entry in metadata.items():
        disaster_type = entry.get('disaster_type')

        if disaster_type not in DISASTER_TYPES:
            invalid_disasters.append((image_id, disaster_type))
            continue

        if disaster_type not in disaster_stats:
            disaster_stats[disaster_type] = {
                'count': 0,
                'severity_sum': 0.0,
                'severity_list': []
            }

        disaster_stats[disaster_type]['count'] += 1
        severity = entry.get('severity', 0.5)
        disaster_stats[disaster_type]['severity_sum'] += severity
        disaster_stats[disaster_type]['severity_list'].append(severity)

    if invalid_disasters:
        print(f"WARNING: Found invalid disaster types")
        for img_id, dtype in invalid_disasters[:5]:
            print(f"  - {img_id}: {dtype}")
        if len(invalid_disasters) > 5:
            print(f"  ... and {len(invalid_disasters) - 5} more")

    print("\nDisaster Type Statistics:")
    total_count = sum(stats['count'] for stats in disaster_stats.values())

    for disaster_id in sorted(disaster_stats.keys()):
        stats = disaster_stats[disaster_id]
        count = stats['count']
        percentage = (count / total_count * 100) if total_count > 0 else 0
        avg_severity = stats['severity_sum'] / count if count > 0 else 0

        name = DISASTER_TYPES.get(disaster_id, f"Unknown({disaster_id})")
        print(f"  {name:20s}: {count:5d} ({percentage:5.1f}%) | Avg Severity: {avg_severity:.3f}")

    print(f"\nTotal Valid Entries: {total_count}\n")

    return {
        'total_entries': len(metadata),
        'valid_entries': total_count,
        'disaster_stats': disaster_stats,
        'invalid_disasters': invalid_disasters,
        'missing_files': missing_files
    }


def generate_plots(metadata_file: str = './data/metadata.json', output_dir: str = './outputs/analysis'):
    """Generate visualization plots with English labels"""

    print("=" * 100)
    print("Generating Visualization Plots")
    print("=" * 100 + "\n")

    # Load metadata
    metadata_path = Path(metadata_file)
    data_dir = metadata_path.parent

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Collect statistics
    disaster_counts = {}
    disaster_severities = {}

    for image_id, entry in metadata.items():
        disaster_type = entry.get('disaster_type')
        if disaster_type not in DISASTER_TYPES:
            continue

        if disaster_type not in disaster_counts:
            disaster_counts[disaster_type] = 0
            disaster_severities[disaster_type] = []

        disaster_counts[disaster_type] += 1
        severity = entry.get('severity', 0.5)
        disaster_severities[disaster_type].append(severity)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Chart 1: Bar chart - Disaster type quantities
    print("[1/4] Generating bar chart...")
    fig, ax = plt.subplots(figsize=(12, 7))

    names = [DISASTER_TYPES[i] for i in sorted(disaster_counts.keys())]
    counts = [disaster_counts[i] for i in sorted(disaster_counts.keys())]
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']

    bars = ax.bar(names, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax.set_title('Disaster Type Distribution - Absolute Count', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    bar_chart_path = output_path / 'disaster_count_bar.png'
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved to {bar_chart_path}")
    plt.close()

    # Chart 2: Pie chart - Disaster type percentages
    print("[2/4] Generating pie chart...")
    fig, ax = plt.subplots(figsize=(10, 8))

    explode = [0.05] * len(names)
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=names,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    # Enhance percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax.set_title('Disaster Type Distribution - Relative Proportion', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    pie_chart_path = output_path / 'disaster_count_pie.png'
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved to {pie_chart_path}")
    plt.close()

    # Chart 3: Combined comparison chart
    print("[3/4] Generating comparison chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Bar chart
    bars = ax1.bar(names, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Absolute Count', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticklabels(names, rotation=15, ha='right')

    # Right: Pie chart
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=names,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax2.set_title('(B) Relative Proportion', fontsize=13, fontweight='bold')

    fig.suptitle('Disaster Data Distribution - Comparative Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    combo_chart_path = output_path / 'disaster_comparison.png'
    plt.savefig(combo_chart_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved to {combo_chart_path}")
    plt.close()

    # Chart 4: Statistics table
    print("[4/4] Generating statistics table...")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    total = sum(counts)
    table_data = []
    table_data.append(['Disaster Type', 'Count', 'Proportion', 'Avg Severity', 'Severity Range'])

    for i, disaster_id in enumerate(sorted(disaster_counts.keys())):
        name = DISASTER_TYPES[disaster_id]
        count = disaster_counts[disaster_id]
        percentage = f"{count/total*100:.1f}%"

        severities = disaster_severities[disaster_id]
        avg_severity = np.mean(severities)
        min_severity = np.min(severities)
        max_severity = np.max(severities)
        severity_range = f"[{min_severity:.2f}, {max_severity:.2f}]"

        table_data.append([
            name,
            str(count),
            percentage,
            f"{avg_severity:.3f}",
            severity_range
        ])

    # Add total row
    table_data.append(['Total', str(total), '100.0%', '-', '-'])

    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.15, 0.15, 0.2, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style total row
    for i in range(5):
        table[(len(table_data)-1, i)].set_facecolor('#FFE66D')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')

    # Alternate row colors
    for i in range(1, len(table_data)-1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    plt.title('Disaster Data - Detailed Statistics Table', fontsize=14, fontweight='bold', pad=20)

    table_path = output_path / 'disaster_statistics_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved to {table_path}")
    plt.close()

    print("\n" + "=" * 100)
    print("All plots generated successfully!")
    print(f"Output directory: {output_path.absolute()}")
    print("=" * 100 + "\n")

    return {
        'bar_chart': bar_chart_path,
        'pie_chart': pie_chart_path,
        'comparison': combo_chart_path,
        'table': table_path
    }


def generate_summary_report(metadata_file: str = './data/metadata.json',
                           output_file: str = './outputs/analysis/metadata_validation_report.txt'):
    """Generate complete validation report in English"""

    print("=" * 100)
    print("Generating Complete Validation Report")
    print("=" * 100 + "\n")

    # Verify data
    verify_result = verify_metadata(metadata_file)

    # Load raw data
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Generate report
    report = []
    report.append("=" * 100)
    report.append("METADATA.JSON VALIDATION REPORT")
    report.append("=" * 100)
    report.append("")

    # Basic information
    report.append("[Basic Information]")
    report.append(f"Verification Time: {Path(metadata_file).stat().st_mtime}")
    report.append(f"metadata.json Path: {Path(metadata_file).absolute()}")
    report.append(f"File Size: {Path(metadata_file).stat().st_size / 1024:.2f} KB")
    report.append(f"Total Entries: {verify_result['total_entries']}")
    report.append(f"Valid Entries: {verify_result['valid_entries']}")
    report.append("")

    # Disaster type statistics
    report.append("[Disaster Type Statistics]")
    total = verify_result['valid_entries']
    for disaster_id in sorted(verify_result['disaster_stats'].keys()):
        stats = verify_result['disaster_stats'][disaster_id]
        count = stats['count']
        percentage = (count / total * 100) if total > 0 else 0
        avg_severity = stats['severity_sum'] / count if count > 0 else 0

        name = DISASTER_TYPES.get(disaster_id, f"Unknown({disaster_id})")
        report.append(f"  {name:20s}: {count:5d} ({percentage:5.1f}%) | Avg Severity: {avg_severity:.3f}")

    report.append("")

    # Data quality
    report.append("[Data Quality Check]")
    if verify_result['invalid_disasters']:
        report.append(f"  Invalid Disaster Types: {len(verify_result['invalid_disasters'])}")
    else:
        report.append(f"  Invalid Disaster Types: 0")

    missing_count = sum(len(v) for v in verify_result['missing_files'].values())
    if missing_count > 0:
        report.append(f"  Missing Files: {missing_count}")
    else:
        report.append(f"  Missing Files: 0")

    report.append("")

    # Conclusion
    report.append("[Conclusion]")
    if verify_result['invalid_disasters'] or sum(len(v) for v in verify_result['missing_files'].values()) > 0:
        report.append("  Status: WARNING - Issues found")
        report.append("  Issues to fix:")
        if verify_result['invalid_disasters']:
            report.append(f"    - Fix {len(verify_result['invalid_disasters'])} invalid disaster types")
        if verify_result['missing_files']['pre']:
            report.append(f"    - Check {len(verify_result['missing_files']['pre'])} missing pre files")
        if verify_result['missing_files']['post']:
            report.append(f"    - Check {len(verify_result['missing_files']['post'])} missing post files")
        if verify_result['missing_files']['mask']:
            report.append(f"    - Check {len(verify_result['missing_files']['mask'])} missing mask files")
    else:
        report.append("  Status: PASSED")
        report.append("  metadata.json data is complete and correct, ready for training")

    report.append("")
    report.append("=" * 100)

    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {output_path.absolute()}")
    print("\n" + '\n'.join(report))


def main():
    """Main function"""
    metadata_file = './data/metadata.json'
    output_dir = './outputs/analysis'

    # Verify metadata.json
    verify_result = verify_metadata(metadata_file)

    if not verify_result:
        print("Verification failed!")
        return 1

    # Generate visualization plots
    try:
        import matplotlib.pyplot as plt
        plot_paths = generate_plots(metadata_file, output_dir)
        print("\nGenerated chart files:")
        for name, path in plot_paths.items():
            print(f"  - {name}: {path}")
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation")
        print("Install with: pip install matplotlib")

    # Generate complete report
    report_file = Path(output_dir) / 'metadata_validation_report.txt'
    generate_summary_report(metadata_file, str(report_file))

    return 0


if __name__ == '__main__':
    sys.exit(main())

