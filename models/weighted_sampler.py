#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加权采样（Weighted Sampling）训练模块

加权采样是一种处理数据不均衡问题的技术，通过为不同类别设置不同的采样权重，
确保在训练过程中每个类别都能得到适当的关注。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional, Dict, Any, List
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightedSamplerConfig:
    """加权采样配置类"""

    def __init__(
        self,
        use_weighted_sampler: bool = True,
        weight_strategy: str = "inverse_frequency",
        custom_weights: Optional[Dict[int, float]] = None,
        temperature: float = 1.0,
        replacement: bool = True
    ):
        """
        初始化加权采样配置

        Args:
            use_weighted_sampler: 是否使用加权采样
            weight_strategy: 权重计算策略
                - "inverse_frequency": 按类别频率的倒数计算权重（推荐）
                - "sqrt_frequency": 按类别频率的平方根倒数计算
                - "custom": 使用自定义权重
                - "balanced": 所有类别等权重
            custom_weights: 自定义权重字典 {class_id: weight}
            temperature: 权重平滑因子 (值越大权重越平衡)
            replacement: 是否有放回采样
        """
        self.use_weighted_sampler = use_weighted_sampler
        self.weight_strategy = weight_strategy
        self.custom_weights = custom_weights or {}
        self.temperature = temperature
        self.replacement = replacement

        logger.info(f"WeightedSamplerConfig initialized:")
        logger.info(f"  - Strategy: {weight_strategy}")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Replacement: {replacement}")


def compute_class_weights(
    class_counts: Dict[int, int],
    strategy: str = "inverse_frequency",
    temperature: float = 1.0,
    custom_weights: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    """
    计算各类别的权重

    Args:
        class_counts: 各类别的样本数量 {class_id: count}
        strategy: 权重计算策略
        temperature: 权重平滑因子
        custom_weights: 自定义权重

    Returns:
        各类别的权重字典 {class_id: weight}
    """

    if strategy == "custom":
        return custom_weights or {}

    if strategy == "balanced":
        return {class_id: 1.0 for class_id in class_counts.keys()}

    total_samples = sum(class_counts.values())
    class_weights = {}

    if strategy == "inverse_frequency":
        # 最常用的方法：权重 = 总样本数 / (类别数 * 该类样本数)
        num_classes = len(class_counts)
        for class_id, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            class_weights[class_id] = weight

    elif strategy == "sqrt_frequency":
        # 平方根策略：权重 = sqrt(总样本数 / (类别数 * 该类样本数))
        num_classes = len(class_counts)
        for class_id, count in class_counts.items():
            weight = np.sqrt(total_samples / (num_classes * count))
            class_weights[class_id] = weight

    else:
        raise ValueError(f"Unknown weight strategy: {strategy}")

    # 应用温度因子进行平滑
    if temperature != 1.0:
        min_weight = min(class_weights.values())
        max_weight = max(class_weights.values())
        for class_id in class_weights:
            # 权重 = (权重 - 最小值)^(1/温度) + 最小值
            normalized = (class_weights[class_id] - min_weight) / (max_weight - min_weight)
            class_weights[class_id] = (normalized ** (1.0 / temperature)) * (max_weight - min_weight) + min_weight

    # 归一化到 [0.5, 2.0] 范围便于理解
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())
    for class_id in class_weights:
        normalized = (class_weights[class_id] - min_weight) / (max_weight - min_weight)
        class_weights[class_id] = 0.5 + normalized * 1.5  # 范围: [0.5, 2.0]

    return class_weights


def create_weighted_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler_config: Optional[WeightedSamplerConfig] = None,
    shuffle: bool = True
) -> DataLoader:
    """
    创建使用加权采样的 DataLoader

    Args:
        dataset: PyTorch Dataset 实例
        batch_size: 批大小
        num_workers: 数据加载工作线程数
        pin_memory: 是否将数据固定在内存中
        sampler_config: 加权采样配置
        shuffle: 是否打乱数据（不使用加权采样时）

    Returns:
        配置好的 DataLoader
    """

    if sampler_config is None:
        sampler_config = WeightedSamplerConfig()

    if not sampler_config.use_weighted_sampler:
        # 不使用加权采样，直接创建普通 DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # 统计各类别样本数
    class_counts = {}
    for sample in dataset:
        disaster_type = sample['disaster_type'].item()
        class_counts[disaster_type] = class_counts.get(disaster_type, 0) + 1

    logger.info(f"Class distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / len(dataset) * 100
        logger.info(f"  Class {class_id}: {count:5d} ({percentage:5.1f}%)")

    # 计算类别权重
    class_weights_dict = compute_class_weights(
        class_counts,
        strategy=sampler_config.weight_strategy,
        temperature=sampler_config.temperature,
        custom_weights=sampler_config.custom_weights
    )

    logger.info(f"Computed class weights ({sampler_config.weight_strategy}):")
    for class_id in sorted(class_weights_dict.keys()):
        weight = class_weights_dict[class_id]
        logger.info(f"  Class {class_id}: {weight:.4f}")

    # 为每个样本分配权重
    sample_weights = []
    for sample in dataset:
        disaster_type = sample['disaster_type'].item()
        weight = class_weights_dict[disaster_type]
        sample_weights.append(weight)

    sample_weights = torch.FloatTensor(sample_weights)

    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=sampler_config.replacement
    )

    # 创建 DataLoader，不设置 shuffle（因为使用了 sampler）
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


class WeightedLossMixin:
    """加权损失函数混合类"""

    def __init__(self, class_weights: Optional[Dict[int, float]] = None):
        """
        初始化加权损失

        Args:
            class_weights: 类别权重字典
        """
        self.class_weights = class_weights or {}

    def get_loss_weight(self, disaster_type: int) -> float:
        """获取指定灾害类型的损失权重"""
        return self.class_weights.get(disaster_type, 1.0)

    def apply_weighted_loss(self, loss: torch.Tensor, batch_disaster_types: torch.Tensor) -> torch.Tensor:
        """
        应用加权损失

        Args:
            loss: 原始损失值 (B,) 或标量
            batch_disaster_types: 批次中的灾害类型 (B,)

        Returns:
            加权后的损失
        """
        if len(loss.shape) == 0:
            # 标量损失，直接返回
            return loss

        # 为每个样本应用权重
        weights = torch.tensor(
            [self.get_loss_weight(dtype.item()) for dtype in batch_disaster_types],
            dtype=loss.dtype,
            device=loss.device
        )

        weighted_loss = (loss * weights).mean()
        return weighted_loss


# ============================================================================
# 加权采样的三种策略详解
# ============================================================================

"""
策略 1: Inverse Frequency (逆频率策略) - 推荐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
权重 = 总样本数 / (类别数 × 该类样本数)

示例（我们的数据）：
  - Volcano: 1056 样本 → 权重 = 3155 / (4 × 1056) ≈ 0.75
  - Earthquake: 1833 样本 → 权重 = 3155 / (4 × 1833) ≈ 0.43
  - Wildfire: 142 样本 → 权重 = 3155 / (4 × 142) ≈ 5.57
  - Flood: 124 样本 → 权重 = 3155 / (4 × 124) ≈ 6.36

优点：
  ✓ 每个类别得到相等的"有效"训练
  ✓ 数据多的类别被欠采样，数据少的类别被过采样
  ✓ 计算简单，结果稳定

缺点：
  ✗ 可能导致多数类欠学习


策略 2: Square Root Frequency (平方根策略)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
权重 = sqrt(总样本数 / (类别数 × 该类样本数))

特点：平衡性介于 Inverse Frequency 和 Balanced 之间
优点：
  ✓ 比 Inverse Frequency 更温和
  ✓ 不会过度欠采样多数类

示例：
  - Volcano: 权重 = sqrt(0.75) ≈ 0.87
  - Earthquake: 权重 = sqrt(0.43) ≈ 0.66
  - Wildfire: 权重 = sqrt(5.57) ≈ 2.36
  - Flood: 权重 = sqrt(6.36) ≈ 2.52


策略 3: Temperature Scaling (温度缩放)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用于调整权重的"平衡度"

Temperature = 1.0 (默认):  使用原始权重
Temperature > 1.0:        权重更平衡（减少差异）
Temperature < 1.0:        权重差异更大（更激进）

示例：
  原始权重:    [0.75, 0.43, 5.57, 6.36]
  T=2.0后:     [1.50, 1.30, 2.05, 2.15]  (更平衡)
  T=0.5后:     [0.10, 0.01, 8.20, 9.15]  (更激进)
"""

# ============================================================================
# 使用示例
# ============================================================================

"""
# 在训练脚本中使用加权采样：

from weighted_sampler import WeightedSamplerConfig, create_weighted_dataloader

# 方案1: 使用逆频率策略（推荐）
sampler_config = WeightedSamplerConfig(
    use_weighted_sampler=True,
    weight_strategy="inverse_frequency",
    temperature=1.0,
    replacement=True
)

# 方案2: 使用平方根策略（更温和）
sampler_config = WeightedSamplerConfig(
    use_weighted_sampler=True,
    weight_strategy="sqrt_frequency",
    temperature=1.0,
    replacement=True
)

# 方案3: 使用自定义权重
sampler_config = WeightedSamplerConfig(
    use_weighted_sampler=True,
    weight_strategy="custom",
    custom_weights={
        0: 1.0,   # Volcano
        1: 0.5,   # Earthquake
        2: 5.0,   # Wildfire
        3: 5.0,   # Flood
    }
)

# 创建 DataLoader
train_dataloader = create_weighted_dataloader(
    dataset=train_dataset,
    batch_size=32,
    num_workers=4,
    sampler_config=sampler_config
)

# 在训练循环中使用
for batch in train_dataloader:
    rgb_image = batch['rgb_image']
    sar_image = batch['sar_image']
    building_mask = batch['building_mask']
    disaster_type = batch['disaster_type']
    
    # 计算损失并反向传播
    ...
"""

