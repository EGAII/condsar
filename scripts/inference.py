"""
CONDSAR 推理脚本
仅使用灾难前光学图像和建筑掩码生成灾难后SAR图像
集成WandB可视化
"""
import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "models"))

from src.utils.wandb_utils import WandBVisualizer
from src.utils.logger import setup_logger
from models.enhanced_condsar import EnhancedDisasterControlNet

logger = logging.getLogger(__name__)


class CondsarInferencer:
    """CONDSAR推理器 - 仅用RGB和mask生成SAR"""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_wandb: bool = True,
        wandb_offline: bool = False,
        output_dir: str = './outputs'
    ):
        """
        初始化推理器

        Args:
            model_path: 训练好的模型检查点路径
            device: 推理设备 ('cuda' 或 'cpu')
            use_wandb: 是否使用WandB记录结果
            wandb_offline: WandB离线模式
            output_dir: 输出目录
        """
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = setup_logger(
            name='condsar_inference',
            log_dir=f"{output_dir}/logs"
        )

        # 初始化WandB
        self.visualizer = None
        if use_wandb:
            try:
                import wandb
                if wandb_offline:
                    os.environ["WANDB_MODE"] = "offline"
                self.visualizer = WandBVisualizer(
                    project_name='condsar-inference',
                    run_name=f"inference_{Path(model_path).stem}"
                )
                self.logger.info("✅ WandB initialized for inference")
            except ImportError:
                self.logger.warning("⚠️ WandB not installed")

        # 加载模型
        self.logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.logger.info(f"✅ Model loaded ({self._count_parameters(self.model):,} parameters)")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        # 创建模型
        model = EnhancedDisasterControlNet(
            num_disaster_types=5,
            embedding_dim=128,
            model_channels=320
        ).to(self.device)

        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            self.logger.info(f"Loaded checkpoint from {model_path}")
        else:
            self.logger.warning(f"Model checkpoint not found at {model_path}, using random initialization")

        model.eval()
        return model

    def load_image(self, image_path: str, target_size: int = 512) -> torch.Tensor:
        """
        加载和预处理图像

        Args:
            image_path: 图像路径
            target_size: 目标大小

        Returns:
            (1, C, H, W) 张量
        """
        img = Image.open(image_path)

        # RGB图像转换
        if img.mode != 'RGB' and img.mode != 'L':
            img = img.convert('RGB')

        # 调整大小
        img = img.resize((target_size, target_size), Image.LANCZOS)

        # 转为张量
        img_array = np.array(img, dtype=np.float32)

        if len(img_array.shape) == 2:  # 灰度图
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        else:  # RGB
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # 归一化
        img_tensor = img_tensor / 255.0
        img_tensor = torch.clamp(img_tensor, 0, 1)

        return img_tensor.to(self.device)

    def infer(
        self,
        rgb_image_path: str,
        mask_path: Optional[str] = None,
        disaster_type: int = 0,
        disaster_severity: float = 0.5,
        num_variants: int = 1,
        output_prefix: str = 'inference'
    ) -> Dict:
        """
        执行推理

        Args:
            rgb_image_path: RGB (灾难前) 图像路径
            mask_path: 建筑掩码路径 (可选)
            disaster_type: 灾难类型 (0-4)
            disaster_severity: 灾难强度 (0-1)
            num_variants: 生成变体数
            output_prefix: 输出文件前缀

        Returns:
            推理结果字典
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔮 Starting Inference")
        self.logger.info("=" * 80)

        self.logger.info(f"📸 Input: {rgb_image_path}")
        if mask_path:
            self.logger.info(f"🎭 Mask: {mask_path}")
        self.logger.info(f"⚡ Disaster Type: {disaster_type}, Severity: {disaster_severity:.2f}")

        # 加载输入
        rgb = self.load_image(rgb_image_path)
        self.logger.info(f"✅ Loaded RGB image: {rgb.shape}")

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = self.load_image(mask_path, target_size=512)
            if mask.shape[1] > 1:
                mask = mask.mean(dim=1, keepdim=True)
            self.logger.info(f"✅ Loaded mask: {mask.shape}")

        results = {
            'input_rgb': rgb_image_path,
            'input_mask': mask_path,
            'disaster_type': disaster_type,
            'disaster_severity': disaster_severity,
            'outputs': []
        }

        # 生成多个变体
        with torch.no_grad():
            for variant_idx in range(num_variants):
                self.logger.info(f"\n📝 Generating variant {variant_idx+1}/{num_variants}...")

                try:
                    # 前向传播
                    output = self.model(
                        sample=torch.randn(1, 1, 512, 512).to(self.device),
                        timestep=torch.tensor([500]).to(self.device),
                        encoder_hidden_states=rgb,
                        rgb_image=rgb,
                        building_mask=mask,
                        disaster_type=torch.tensor([disaster_type]).to(self.device),
                        disaster_severity=torch.tensor([disaster_severity]).to(self.device)
                    )

                    # 限制输出范围
                    output = torch.clamp(output, 0, 1)

                    self.logger.info(f"✅ Generated SAR: {output.shape}, "
                                   f"range: [{output.min():.3f}, {output.max():.3f}]")

                    # 保存输出
                    output_path = self.output_dir / f"{output_prefix}_var{variant_idx+1}.png"
                    self._save_image(output, str(output_path))
                    self.logger.info(f"💾 Saved to {output_path}")

                    results['outputs'].append({
                        'path': str(output_path),
                        'shape': output.shape,
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.mean().item()
                    })

                    # WandB可视化
                    if self.visualizer:
                        self.visualizer.log_inference_results(
                            f"result_{variant_idx+1}",
                            rgb=rgb,
                            sar_generated=output,
                            mask=mask,
                            disaster_info={
                                'type': disaster_type,
                                'severity': disaster_severity
                            }
                        )

                        # 记录特征图
                        self.visualizer.log_feature_map(
                            f"sar_heatmap_{variant_idx+1}",
                            features=output,
                            stage="inference"
                        )

                except Exception as e:
                    self.logger.error(f"❌ Error in variant {variant_idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 保存结果摘要
        summary_path = self.output_dir / f"{output_prefix}_summary.json"
        with open(summary_path, 'w') as f:
            # 转换为可序列化格式
            summary = {
                'input': results['input_rgb'],
                'mask': results['input_mask'],
                'disaster_type': results['disaster_type'],
                'disaster_severity': results['disaster_severity'],
                'num_variants': len(results['outputs']),
                'outputs': results['outputs']
            }
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n✅ Inference completed")
        self.logger.info(f"💾 Summary saved to {summary_path}")

        if self.visualizer:
            self.visualizer.finish()

        return results

    def batch_infer(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        disaster_type: int = 0,
        disaster_severity: float = 0.5,
        output_prefix: str = 'batch'
    ) -> Dict:
        """
        批量推理

        Args:
            image_dir: RGB图像目录
            mask_dir: 建筑掩码目录 (可选)
            disaster_type: 灾难类型
            disaster_severity: 灾难强度
            output_prefix: 输出前缀

        Returns:
            批量推理结果
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 Batch Inference")
        self.logger.info("=" * 80)

        image_paths = sorted(Path(image_dir).glob('*.jpg')) + \
                     sorted(Path(image_dir).glob('*.png'))

        self.logger.info(f"Found {len(image_paths)} images")

        batch_results = {
            'total_images': len(image_paths),
            'disaster_type': disaster_type,
            'disaster_severity': disaster_severity,
            'results': []
        }

        for idx, img_path in enumerate(image_paths, 1):
            self.logger.info(f"\n[{idx}/{len(image_paths)}] Processing {img_path.name}")

            # 对应的mask
            mask_path = None
            if mask_dir:
                mask_path = Path(mask_dir) / img_path.name
                if not mask_path.exists():
                    mask_path = None

            # 推理
            result = self.infer(
                str(img_path),
                str(mask_path) if mask_path else None,
                disaster_type=disaster_type,
                disaster_severity=disaster_severity,
                output_prefix=f"{output_prefix}_{img_path.stem}"
            )

            batch_results['results'].append(result)

        # 保存批量结果
        batch_summary_path = self.output_dir / f"{output_prefix}_batch_summary.json"
        with open(batch_summary_path, 'w') as f:
            summary = {
                'total_images': batch_results['total_images'],
                'disaster_type': batch_results['disaster_type'],
                'disaster_severity': batch_results['disaster_severity'],
                'processed': len(batch_results['results'])
            }
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n✅ Batch inference completed")
        self.logger.info(f"📊 Summary: {len(batch_results['results'])}/{len(image_paths)} processed")
        self.logger.info(f"💾 Batch summary saved to {batch_summary_path}")

        return batch_results

    def _save_image(self, tensor: torch.Tensor, path: str):
        """保存图像"""
        # 转为numpy
        img = tensor.detach().cpu().numpy()

        # 处理形状
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]

        # 转为8位
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 保存
        Image.fromarray(img).save(path)

    def _count_parameters(self, model: nn.Module) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='CONDSAR Inference Script')

    parser.add_argument('--model', type=str, default='./outputs/checkpoints/best_model.pt',
                       help='Model checkpoint path')
    parser.add_argument('--image', type=str, default='./data/target/pre/image_001.jpg',
                       help='Input RGB image path')
    parser.add_argument('--mask', type=str, default=None,
                       help='Input mask path (optional)')
    parser.add_argument('--disaster-type', type=int, default=0,
                       help='Disaster type (0-4: volcano, earthquake, wildfire, storm, flood)')
    parser.add_argument('--severity', type=float, default=0.5,
                       help='Disaster severity (0-1)')
    parser.add_argument('--batch-dir', type=str, default=None,
                       help='Batch inference image directory')
    parser.add_argument('--mask-dir', type=str, default=None,
                       help='Batch inference mask directory')
    parser.add_argument('--num-variants', type=int, default=1,
                       help='Number of output variants')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./outputs/inference',
                       help='Output directory')
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Use WandB for logging')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run WandB in offline mode')

    args = parser.parse_args()

    # 创建推理器
    inferencer = CondsarInferencer(
        model_path=args.model,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_offline=args.wandb_offline,
        output_dir=args.output_dir
    )

    # 执行推理
    if args.batch_dir:
        inferencer.batch_infer(
            image_dir=args.batch_dir,
            mask_dir=args.mask_dir,
            disaster_type=args.disaster_type,
            disaster_severity=args.severity,
            output_prefix='batch'
        )
    else:
        inferencer.infer(
            rgb_image_path=args.image,
            mask_path=args.mask,
            disaster_type=args.disaster_type,
            disaster_severity=args.severity,
            num_variants=args.num_variants,
            output_prefix='inference'
        )


if __name__ == '__main__':
    main()

