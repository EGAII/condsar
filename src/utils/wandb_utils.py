"""
WandB Integration and Visualization Utilities
集成WandB用于模型训练的可视化和可追踪性
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import wandb
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WandBVisualizer:
    """WandB 可视化工具"""

    def __init__(self, project_name: str = "condsar", run_name: str = None):
        self.project_name = project_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化WandB
        wandb.init(
            project=project_name,
            name=self.run_name,
            reinit=True
        )

    def log_metrics(self, metrics: Dict, step: int = None, stage: str = ""):
        """记录标量指标"""
        wandb_metrics = {f"{stage}/{k}" if stage else k: v for k, v in metrics.items()}
        wandb.log(wandb_metrics, step=step)

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: int = None,
        stage: str = "",
        normalize: bool = True
    ):
        """记录单张图像

        Args:
            name: 图像名称
            image: 张量 (C,H,W) 或 (H,W)
            step: 训练步数
            stage: 阶段标记 (e.g., "stage_a", "inference")
            normalize: 是否归一化到[0,1]
        """
        # 转为numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # 处理形状
        if image.ndim == 3:
            # (C,H,W) -> (H,W,C)
            if image.shape[0] == 1:
                image = image[0]  # 灰度图
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

        # 归一化
        if normalize:
            if image.min() < 0:
                image = (image + 1) / 2
            elif image.max() > 1:
                image = image / 255.0

        # 记录
        wandb_image = wandb.Image(image, caption=name)
        full_name = f"{stage}/{name}" if stage else name
        wandb.log({full_name: wandb_image}, step=step)

    def log_image_grid(
        self,
        name: str,
        images: List[torch.Tensor],
        step: int = None,
        stage: str = "",
        nrow: int = 4
    ):
        """记录图像网格"""
        if isinstance(images, torch.Tensor):
            grid = images
        else:
            grid = torch.stack(images, dim=0)

        # 创建网格
        if grid.ndim == 3:
            grid = grid.unsqueeze(1)

        grid_img = F.grid(grid, nrow=nrow)
        self.log_image(name, grid_img, step=step, stage=stage)

    def log_feature_map(
        self,
        name: str,
        features: torch.Tensor,
        step: int = None,
        stage: str = "",
        channel_idx: List[int] = None,
        max_channels: int = 16
    ):
        """记录特征图

        Args:
            name: 特征图名称
            features: (B,C,H,W) 张量
            step: 训练步数
            stage: 阶段标记
            channel_idx: 要显示的通道索引，None则自动选择
            max_channels: 最多显示的通道数
        """
        features = features.detach().cpu()

        if features.ndim != 4:
            logger.warning(f"Expected 4D tensor, got {features.ndim}D")
            return

        B, C, H, W = features.shape

        # 选择通道
        if channel_idx is None:
            # 自动选择均匀分布的通道
            step_size = max(1, C // max_channels)
            channel_idx = list(range(0, C, step_size))[:max_channels]

        # 取第一个batch的特征
        feature_maps = features[0, channel_idx, :, :]

        # 可视化
        num_channels = len(channel_idx)
        ncols = min(4, num_channels)
        nrows = (num_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
        axes = axes.flatten() if num_channels > 1 else [axes]

        for idx, (ax, channel_id) in enumerate(zip(axes, channel_idx)):
            feature_map = feature_maps[idx].numpy()

            # 归一化
            fmin = feature_map.min()
            fmax = feature_map.max()
            if fmax > fmin:
                feature_map = (feature_map - fmin) / (fmax - fmin)

            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Channel {channel_id}')
            ax.axis('off')

        # 隐藏多余的子图
        for idx in range(num_channels, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # 记录
        full_name = f"{stage}/{name}" if stage else name
        wandb.log({full_name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_training_comparison(
        self,
        name: str,
        rgb: torch.Tensor,
        sar_pred: torch.Tensor,
        sar_gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        step: int = None,
        stage: str = ""
    ):
        """记录训练对比图：RGB | SAR预测 | SAR真值 | Mask

        Args:
            name: 对比图名称
            rgb: RGB图像 (1,3,H,W)
            sar_pred: 预测的SAR (1,1,H,W)
            sar_gt: 真值SAR (1,1,H,W)
            mask: 掩码 (1,1,H,W)
            step: 训练步数
            stage: 阶段标记
        """
        fig, axes = plt.subplots(1, 4 if sar_gt is not None else 3, figsize=(16, 4))

        # RGB
        rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        rgb_np = np.clip(rgb_np, 0, 1)
        axes[0].imshow(rgb_np)
        axes[0].set_title('RGB Pre-disaster')
        axes[0].axis('off')

        # SAR预测
        sar_pred_np = sar_pred[0, 0].detach().cpu().numpy()
        axes[1].imshow(sar_pred_np, cmap='gray')
        axes[1].set_title('SAR Predicted')
        axes[1].axis('off')

        # SAR真值
        if sar_gt is not None:
            sar_gt_np = sar_gt[0, 0].detach().cpu().numpy()
            axes[2].imshow(sar_gt_np, cmap='gray')
            axes[2].set_title('SAR Ground Truth')
            axes[2].axis('off')

            # Mask
            if mask is not None:
                mask_np = mask[0, 0].detach().cpu().numpy()
                axes[3].imshow(mask_np, cmap='binary')
                axes[3].set_title('Building Mask')
                axes[3].axis('off')

        plt.tight_layout()

        # 记录
        full_name = f"{stage}/{name}" if stage else name
        wandb.log({full_name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_inference_results(
        self,
        name: str,
        rgb: torch.Tensor,
        sar_generated: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        step: int = None,
        disaster_info: Dict = None
    ):
        """记录推理结果"""
        fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(12, 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        # RGB
        rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().numpy() if rgb.ndim == 4 else rgb
        rgb_np = np.clip(rgb_np, 0, 1)
        axes[0].imshow(rgb_np)
        axes[0].set_title('Input RGB Image')
        axes[0].axis('off')

        # 生成的SAR
        sar_np = sar_generated[0, 0].detach().cpu().numpy() if sar_generated.ndim == 4 else sar_generated[0]
        axes[1].imshow(sar_np, cmap='gray')
        title = 'Generated SAR'
        if disaster_info:
            title += f"\n(Type: {disaster_info.get('type', 'N/A')}, Severity: {disaster_info.get('severity', 'N/A'):.2f})"
        axes[1].set_title(title)
        axes[1].axis('off')

        # Mask
        if mask is not None:
            mask_np = mask[0, 0].detach().cpu().numpy() if mask.ndim == 4 else mask[0]
            axes[2].imshow(mask_np, cmap='binary')
            axes[2].set_title('Building Mask')
            axes[2].axis('off')

        plt.tight_layout()

        # 记录
        wandb.log({f"inference/{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: int = None,
        stage: str = "",
        bins: int = 50
    ):
        """记录直方图"""
        values = values.detach().cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(name)
        plt.tight_layout()

        full_name = f"{stage}/{name}" if stage else name
        wandb.log({full_name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def finish(self):
        """完成WandB记录"""
        wandb.finish()


class VisualizationCallback:
    """训练回调：用于可视化"""

    def __init__(self, visualizer: WandBVisualizer, log_frequency: int = 100):
        self.visualizer = visualizer
        self.log_frequency = log_frequency

    def on_train_batch_end(
        self,
        batch_idx: int,
        batch: Dict,
        outputs: Dict,
        stage: str = "stage_a"
    ):
        """训练batch结束回调"""
        if batch_idx % self.log_frequency != 0:
            return

        # 记录指标
        if 'loss' in outputs:
            self.visualizer.log_metrics(
                {'loss': outputs['loss']},
                step=batch_idx,
                stage=stage
            )

        # 记录图像对比
        if 'rgb' in batch and 'sar_pred' in outputs:
            self.visualizer.log_training_comparison(
                f'batch_{batch_idx:06d}',
                rgb=batch['rgb'][:1],  # 只显示第一张
                sar_pred=outputs['sar_pred'][:1],
                sar_gt=batch.get('sar')[:1] if 'sar' in batch else None,
                mask=batch.get('mask')[:1] if 'mask' in batch else None,
                step=batch_idx,
                stage=stage
            )

        # 记录特征图
        if 'features' in outputs:
            self.visualizer.log_feature_map(
                f'feature_map_{batch_idx:06d}',
                features=outputs['features'][:1],
                step=batch_idx,
                stage=stage,
                max_channels=16
            )

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict,
        stage: str = "stage_a"
    ):
        """epoch结束回调"""
        self.visualizer.log_metrics(metrics, step=epoch, stage=f"{stage}/epoch")


def create_wandb_config() -> Dict:
    """创建WandB配置"""
    return {
        "project": "condsar",
        "entity": None,
        "tags": ["disaster-sar", "controlnet"],
        "notes": "CONDSAR: Disaster-Aware SAR Image Generation",
    }


if __name__ == "__main__":
    # 测试
    viz = WandBVisualizer("condsar-test")

    # 测试图像
    rgb = torch.rand(1, 3, 256, 256)
    sar = torch.rand(1, 1, 256, 256)
    features = torch.rand(1, 64, 32, 32)

    viz.log_image("test_rgb", rgb[0])
    viz.log_feature_map("test_features", features)
    viz.log_training_comparison(
        "test_comparison",
        rgb=rgb,
        sar_pred=sar,
        sar_gt=sar + 0.1 * torch.randn_like(sar)
    )

    viz.finish()
    print("✅ WandB visualization test completed")

