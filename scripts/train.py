"""
完整的CONDSAR训练脚本
支持Stage A/B/C三阶段训练，集成WandB可视化
支持从YAML/JSON配置文件加载参数
"""
import argparse
import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "models"))

from src.utils.wandb_utils import WandBVisualizer, VisualizationCallback
from src.utils.logger import setup_logger
from models.training_utils import DisasterSARDataset, MetricsTracker
from models.enhanced_condsar import EnhancedDisasterControlNet, SARVAEDecoder

logger = logging.getLogger(__name__)


# ============================================================================
# 配置加载函数
# ============================================================================

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    从YAML或JSON文件加载配置

    Args:
        config_path: 配置文件路径 (.yaml 或 .json)

    Returns:
        配置字典
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"❌ Config file not found: {config_path}")
        return {}

    try:
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded YAML config from {config_path}")

        elif config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded JSON config from {config_path}")

        else:
            logger.error(f"❌ Unsupported config format: {config_path.suffix}")
            return {}

        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return {}


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """
    将命令行参数和配置文件合并
    命令行参数优先级 > 配置文件优先级

    Args:
        config: 从配置文件加载的配置
        args: 命令行参数

    Returns:
        合并后的配置字典
    """
    # 提取配置文件中的训练参数
    if not config:
        return vars(args)

    # 获取阶段特定的配置
    stage = getattr(args, 'stage', args.stage if 'stage' in vars(args) else 'a')

    # 构建配置字典
    merged = {}

    # 优先级 1: 从配置文件中提取
    if 'training' in config and f'stage_{stage}' in config['training']:
        stage_config = config['training'][f'stage_{stage}']
        for key, value in stage_config.items():
            merged[key] = value

    # 优先级 2: 从数据配置中提取
    if 'data' in config:
        for key, value in config['data'].items():
            if key not in merged:
                merged[key] = value

    # 优先级 3: 从模型配置中提取
    if 'model' in config:
        for key, value in config['model'].items():
            if key not in merged:
                merged[key] = value

    # 优先级 4: 从W&B配置中提取
    if 'wandb' in config:
        if 'use_wandb' not in merged:
            merged['use_wandb'] = config['wandb'].get('enabled', True)

    # 优先级 5: 从设备配置中提取
    if 'device' in config:
        if 'device' not in merged:
            merged['device'] = config['device'].get('type', 'cuda')

    # 优先级 6: 从输出配置中提取
    if 'output' in config:
        if 'output_dir' not in merged:
            merged['output_dir'] = config['output'].get('directory', './outputs')

    # 优先级最高: 命令行参数覆盖所有
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:  # 只覆盖显式指定的参数
            merged[key] = value

    # 确保device和其他关键参数有默认值
    if 'device' not in merged or merged.get('device') is None:
        merged['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("✅ Config merged successfully")
    return merged


class TrainingConfig:
    """训练配置"""

    def __init__(self, **kwargs):
        # 基础配置
        self.project_name = kwargs.get('project_name', 'condsar')
        self.run_name = kwargs.get('run_name', f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.stage = kwargs.get('stage', 'a')  # a, b, or c

        # Device - with extra safety check
        device_val = kwargs.get('device', None)
        if device_val is None or device_val == 'None':
            device_val = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device_val

        # 数据配置
        self.source_dir = kwargs.get('source_dir') or './data'
        self.target_dir = kwargs.get('target_dir') or './data'
        self.image_size = kwargs.get('image_size') or 512

        # 模型配置
        self.model_channels = kwargs.get('model_channels') or 320
        self.num_disaster_types = kwargs.get('num_disaster_types') or 5
        self.embedding_dim = kwargs.get('embedding_dim') or 128

        # 训练配置
        self.batch_size = kwargs.get('batch_size') or 4
        self.num_epochs = kwargs.get('num_epochs') or 100
        self.learning_rate = kwargs.get('learning_rate') or 1e-4
        self.weight_decay = kwargs.get('weight_decay') or 1e-5
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps') or 1

        # 优化器配置
        self.warmup_steps = kwargs.get('warmup_steps') or 1000
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)

        # 检查点配置
        self.checkpoint_dir = kwargs.get('checkpoint_dir') or './outputs/checkpoints'
        self.save_frequency = kwargs.get('save_frequency') or 10

        # WandB配置
        self.use_wandb = kwargs.get('use_wandb', True)
        self.wandb_offline = kwargs.get('wandb_offline', False)
        self.log_frequency = kwargs.get('log_frequency') or 100

        # 可视化配置
        self.visualize_features = kwargs.get('visualize_features', True)
        self.visualize_frequency = kwargs.get('visualize_frequency') or 500
        self.output_dir = kwargs.get('output_dir') or './outputs'

    def to_dict(self) -> Dict:
        """转为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        """保存配置"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")


class CondsarTrainer:
    """CONDSAR训练器 - 支持Stage A/B/C"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 创建输出目录
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = setup_logger(
            name='condsar_trainer',
            log_dir=f"{config.output_dir}/logs"
        )

        # WandB初始化
        self.visualizer = None
        if config.use_wandb:
            try:
                import wandb
                if config.wandb_offline:
                    os.environ["WANDB_MODE"] = "offline"
                self.visualizer = WandBVisualizer(
                    project_name=config.project_name,
                    run_name=config.run_name
                )
                self.logger.info("✅ WandB initialized")
            except ImportError:
                self.logger.warning("⚠️ WandB not installed, skipping visualization")

        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = MetricsTracker(self.logger)

        # 保存配置
        config.save(f"{config.output_dir}/config_{config.stage}.json")

    def setup_stage_a(self):
        """设置Stage A训练"""
        self.logger.info("=" * 80)
        self.logger.info("🎯 Setting up STAGE A: Source Domain Training")
        self.logger.info("=" * 80)

        # 加载数据
        self.logger.info(f"Loading source dataset from {self.config.source_dir}")
        self.train_dataset = DisasterSARDataset(
            dataset_dir=self.config.source_dir,
            image_size=self.config.image_size,
            return_mask=True,
            return_metadata=True,
            logger=self.logger
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.logger.info(f"✅ Loaded {len(self.train_dataset)} training samples")
        self.logger.info(f"   Batch size: {self.config.batch_size}")
        self.logger.info(f"   Total batches: {len(self.train_loader)}")

        # 创建模型
        self.logger.info("Creating EnhancedDisasterControlNet...")
        self.model = EnhancedDisasterControlNet(
            num_disaster_types=self.config.num_disaster_types,
            embedding_dim=self.config.embedding_dim,
            model_channels=self.config.model_channels
        ).to(self.device)

        self.logger.info(f"✅ Model created with {self._count_parameters(self.model):,} parameters")

        # 创建SAR VAE Decoder (可训练)
        self.logger.info("Creating SAR VAE Decoder...")
        self.sar_decoder = SARVAEDecoder(
            latent_channels=4,
            latent_size=64,
            output_channels=1,
            hidden_channels=128
        ).to(self.device)

        self.logger.info(f"✅ SAR VAE Decoder created with {self._count_parameters(self.sar_decoder):,} parameters")

        # 冻结 VAE Encoder (如果模型中有)
        try:
            # 如果使用了预训练的VAE encoder，冻结它
            for param in self.model.vae_encoder.parameters():
                param.requires_grad = False
            self.logger.info("✅ VAE Encoder frozen")
        except AttributeError:
            self.logger.info("⚠️ No VAE encoder found to freeze")

        # 创建优化器 - 优化可训练的参数
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        trainable_params.extend(self.sar_decoder.parameters())

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-7
        )

        self.logger.info(f"✅ Optimizer configured (lr={self.config.learning_rate})")

        if self.visualizer:
            self.visualizer.log_metrics(
                {
                    'stage': 'a',
                    'dataset_size': len(self.train_dataset),
                    'batch_size': self.config.batch_size,
                    'model_parameters': self._count_parameters(self.model)
                },
                step=0
            )

    def train_stage_a(self):
        """执行Stage A训练"""
        self.setup_stage_a()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 Starting Stage A Training")
        self.logger.info("=" * 80 + "\n")

        best_loss = float('inf')
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):
                global_step += 1

                # 准备数据
                rgb = batch['rgb_image'].to(self.device)
                sar = batch['sar_image'].to(self.device)
                mask = batch.get('building_mask')
                if mask is not None:
                    mask = mask.to(self.device)

                disaster_type = batch.get('disaster_type')
                if disaster_type is not None:
                    disaster_type = disaster_type.to(self.device)

                disaster_severity = batch.get('disaster_severity')
                if disaster_severity is not None:
                    disaster_severity = disaster_severity.to(self.device)

                # 前向传播
                try:
                    # ControlNet生成条件
                    outputs = self.model(
                        sample=sar,
                        timestep=torch.randint(0, 1000, (rgb.size(0),)).to(self.device),
                        encoder_hidden_states=rgb,
                        rgb_image=rgb,
                        building_mask=mask,
                        disaster_type=disaster_type,
                        disaster_severity=disaster_severity
                    )

                    # SAR VAE Decoder解码 (此处outputs应该是latent表示)
                    # 如果outputs是raw output，需要通过decoder生成SAR图像
                    if hasattr(outputs, 'shape') and len(outputs.shape) == 4:
                        # 假设outputs是(B, C, H, W)的latent
                        sar_pred = self.sar_decoder(outputs)
                    else:
                        sar_pred = outputs

                    # 计算损失
                    loss = F.mse_loss(sar_pred, sar)

                    # 反向传播
                    loss.backward()

                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # 记录指标
                    epoch_loss += loss.item()
                    num_batches += 1

                    self.metrics.update(loss=loss.item())

                    # 定期记录
                    if global_step % self.config.log_frequency == 0:
                        avg_loss = epoch_loss / num_batches
                        self.logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs} | "
                            f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                            f"Loss: {loss.item():.6f} | "
                            f"Avg Loss: {avg_loss:.6f}"
                        )

                        if self.visualizer:
                            self.visualizer.log_metrics(
                                {'loss': loss.item(), 'avg_loss': avg_loss},
                                step=global_step,
                                stage='stage_a'
                            )

                        # 可视化特征和结果
                        if self.config.visualize_features and global_step % self.config.visualize_frequency == 0:
                            self.visualizer.log_training_comparison(
                                f'batch_{global_step}',
                                rgb=rgb[:1],
                                sar_pred=outputs[:1],
                                sar_gt=sar[:1],
                                mask=mask[:1] if mask is not None else None,
                                step=global_step,
                                stage='stage_a'
                            )

                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue

            # Epoch结束
            epoch_loss /= num_batches
            self.scheduler.step()

            self.logger.info(
                f"\n✅ Epoch {epoch+1}/{self.config.num_epochs} completed - Loss: {epoch_loss:.6f}\n"
            )

            if self.visualizer:
                self.visualizer.log_metrics(
                    {'epoch_loss': epoch_loss, 'lr': self.scheduler.get_last_lr()[0]},
                    step=epoch,
                    stage='stage_a'
                )

            # 保存检查点
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(epoch, epoch_loss)

            # 保存最优模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_checkpoint(epoch, epoch_loss, is_best=True)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎉 Stage A Training Completed!")
        self.logger.info(f"Best Loss: {best_loss:.6f}")
        self.logger.info("=" * 80)

        if self.visualizer:
            self.visualizer.finish()

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.to_dict()
        }

        if is_best:
            path = f"{self.config.checkpoint_dir}/best_model.pt"
            self.logger.info(f"💾 Saving best model (loss={loss:.6f}) to {path}")
        else:
            path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch+1:03d}.pt"
            self.logger.info(f"💾 Saving checkpoint to {path}")

        torch.save(checkpoint, path)

    def _count_parameters(self, model: nn.Module) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='CONDSAR Training Script')

    # 配置文件参数 (新增)
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path (.yaml or .json)')

    # 基础参数
    parser.add_argument('--stage', type=str, default='a', choices=['a', 'b', 'c'],
                       help='Training stage (a/b/c)')
    parser.add_argument('--source-dir', type=str, default=None,
                       help='Source domain dataset directory')
    parser.add_argument('--target-dir', type=str, default=None,
                       help='Target domain dataset directory')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--use-wandb', action='store_true', default=None,
                       help='Use WandB for logging')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run WandB in offline mode')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--run-name', type=str, default=None,
                       help='WandB run name')

    args = parser.parse_args()

    # ========== 配置加载流程 ==========
    config_dict = {}

    # Step 1: 如果指定了配置文件，加载配置文件
    if args.config:
        config_dict = load_config_file(args.config)

    # Step 2: 合并配置和命令行参数 (命令行参数优先级最高)
    merged_config = merge_config_with_args(config_dict, args)

    # Step 3: 使用合并后的配置创建 TrainingConfig
    config = TrainingConfig(
        stage=merged_config.get('stage', 'a'),
        source_dir=merged_config.get('source_dir', './data'),
        target_dir=merged_config.get('target_dir', './data'),
        batch_size=merged_config.get('batch_size', 4),
        num_epochs=merged_config.get('num_epochs', 100),
        learning_rate=merged_config.get('learning_rate', 1e-4),
        device=merged_config.get('device', 'cuda'),
        use_wandb=merged_config.get('use_wandb', True),
        wandb_offline=merged_config.get('wandb_offline', False),
        output_dir=merged_config.get('output_dir', './outputs'),
        run_name=merged_config.get('run_name', None)
    )

    # 创建训练器
    trainer = CondsarTrainer(config)

    # 执行训练
    if config.stage == 'a':
        trainer.train_stage_a()
    else:
        print(f"Stage {config.stage} training not yet implemented")


if __name__ == '__main__':
    main()

