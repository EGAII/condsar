"""
Training Utilities for CONDSAR
Includes logger setup, data utilities, W&B integration, and metrics tracking
"""
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import tifffile

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # Set to None when not available
    print("Warning: wandb not installed. Install with: pip install wandb")

# For type checking without runtime errors
if TYPE_CHECKING:
    import wandb as wandb_types


class MaskEncoder(nn.Module):
    """
    Mask编码器：将灾害损伤mask (0-3) 编码为embedding
    参考NEDS的设计，可训练

    0: 背景
    1: 完好
    2: 轻度损伤
    3: 重度损伤/摧毁
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 128,
        num_damage_levels: int = 4,
        hidden_channels: int = 64
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_damage_levels = num_damage_levels

        # 损伤等级的learnable嵌入表
        self.damage_level_embedding = nn.Embedding(
            num_embeddings=num_damage_levels,
            embedding_dim=embedding_dim
        )

        # Mask编码器网络 - 用于提取空间特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # MLP投影层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            mask: (B, 1, H, W) 损伤mask [0, 1, 2, 3]

        Returns:
            damage_embedding: (B, embedding_dim) 融合的embedding
        """
        B = mask.shape[0]

        # 编码mask的空间特征
        features = self.encoder(mask)
        features_flat = features.view(B, -1)

        # MLP投影
        spatial_embedding = self.mlp(features_flat)

        # 生成离散化的damage embedding
        mask_discrete = torch.clamp(mask.long(), 0, self.num_damage_levels - 1)
        mask_flat = mask_discrete.view(B, -1)

        # 计算每个等级的出现比例
        damage_embeddings = []
        for level in range(self.num_damage_levels):
            level_mask = (mask_flat == level).float()
            level_weight = level_mask.sum(dim=1, keepdim=True) / (mask.size(-1) * mask.size(-2))

            level_embedding = self.damage_level_embedding(
                torch.tensor([level], device=mask.device, dtype=torch.long)
            )
            damage_embeddings.append((level_embedding * level_weight.unsqueeze(-1)).squeeze(0))

        # 加权求和
        discrete_embedding = torch.stack(damage_embeddings, dim=0).sum(dim=0)

        # 融合空间特征和离散embedding
        fused_embedding = spatial_embedding + discrete_embedding

        return fused_embedding


class ColoredFormatter(logging.Formatter):
    """
    Mask编码器：将灾害损伤mask (0-3) 编码为embedding
    参考NEDS的设计

    0: 背景
    1: 完好
    2: 轻度损伤
    3: 重度损伤/摧毁
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 128,
        num_damage_levels: int = 4,
        hidden_channels: int = 64
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_damage_levels = num_damage_levels

        # 损伤等级的learnable嵌入表
        self.damage_level_embedding = nn.Embedding(
            num_embeddings=num_damage_levels,
            embedding_dim=embedding_dim
        )

        # Mask编码器网络 - 用于提取空间特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # MLP投影层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            mask: (B, 1, H, W) 损伤mask [0, 1, 2, 3]

        Returns:
            damage_embedding: (B, embedding_dim) 融合的embedding
        """
        B = mask.shape[0]

        # 编码mask的空间特征
        features = self.encoder(mask)
        features_flat = features.view(B, -1)

        # MLP投影
        spatial_embedding = self.mlp(features_flat)

        # 生成离散化的damage embedding
        mask_discrete = torch.clamp(mask.long(), 0, self.num_damage_levels - 1)
        mask_flat = mask_discrete.view(B, -1)

        # 计算每个等级的出现比例
        damage_embeddings = []
        for level in range(self.num_damage_levels):
            level_mask = (mask_flat == level).float()
            level_weight = level_mask.sum(dim=1, keepdim=True) / (mask.size(-1) * mask.size(-2))

            level_embedding = self.damage_level_embedding(
                torch.tensor([level], device=mask.device, dtype=torch.long)
            )
            damage_embeddings.append((level_embedding * level_weight.unsqueeze(-1)).squeeze(0))

        # 加权求和
        discrete_embedding = torch.stack(damage_embeddings, dim=0).sum(dim=0)

        # 融合空间特征和离散embedding
        fused_embedding = spatial_embedding + discrete_embedding

        return fused_embedding


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_dir: str = "./logs",
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup logger with both console and file handlers

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to write to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Format
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_wandb(
    project_name: str = "condsar",
    run_name: str = None,
    config: Dict[str, Any] = None,
    offline: bool = False,
    disabled: bool = not HAS_WANDB
) -> Optional["wandb_types.run"]:
    """
    Initialize Weights & Biases logging

    Args:
        project_name: W&B project name
        run_name: Name for this run (auto-generated if None)
        config: Configuration dictionary to log
        offline: Run in offline mode
        disabled: Disable W&B entirely

    Returns:
        W&B run object or None
    """
    if disabled:
        print("W&B logging disabled")
        return None

    if not HAS_WANDB:
        print("Warning: wandb not installed, skipping W&B initialization")
        return None

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"condsar_{timestamp}"

    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        mode="offline" if offline else "online"
    )

    return run


class DisasterSARDataset(Dataset):
    """
    Dataset for Disaster SAR Image Generation

    Expected structure with metadata.json:
    dataset/
    ├── metadata.json  # File paths and disaster info
    ├── pre/           # Pre-disaster RGB images
    ├── post/          # Post-disaster SAR images
    └── mask/          # Building/damage masks

    metadata.json format (REQUIRED):
    {
        "image_id": {
            "pre": "path/to/pre/image.jpg",
            "post": "path/to/post/image.tif",
            "mask": "path/to/mask/image.tif",
            "disaster_type": 0-3,  # 0=Volcano, 1=Earthquake, 2=Wildfire, 3=Flood
            "severity": 0.0-1.0    # Disaster intensity
        }
    }
    """

    DISASTER_TYPES = {
        0: "Volcano",
        1: "Earthquake",
        2: "Wildfire",
        3: "Flood"
    }

    def __init__(
        self,
        dataset_dir: str,
        image_size: int = 512,
        return_mask: bool = True,
        return_metadata: bool = True,
        logger: logging.Logger = None
    ):
        """
        Args:
            dataset_dir: Path to dataset directory (must contain metadata.json)
            image_size: Size to resize images to
            return_mask: Whether to return building mask
            return_metadata: Whether to return disaster type and severity
            logger: Logger instance
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.return_mask = return_mask
        self.return_metadata = return_metadata
        self.logger = logger or logging.getLogger(__name__)

        # Load metadata (REQUIRED)
        self.metadata = {}
        metadata_file = self.dataset_dir / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.dataset_dir}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.image_ids = sorted(list(self.metadata.keys()))

        self.logger.info(f"Loaded {len(self.image_ids)} images from metadata.json")
        self.logger.info(f"Disaster types: {list(self.DISASTER_TYPES.values())}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        meta_entry = self.metadata[image_id]

        # Get file paths from metadata
        pre_path = self.dataset_dir / meta_entry['pre']
        post_path = self.dataset_dir / meta_entry['post']
        mask_path = self.dataset_dir / meta_entry['mask'] if self.return_mask else None

        # Load RGB pre-disaster image
        if not pre_path.exists():
            raise FileNotFoundError(f"RGB image not found: {pre_path}")

        rgb_image = self._load_image(pre_path, mode='RGB')
        rgb_image = rgb_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float() / 255.0

        # Load SAR post-disaster image
        sar_tensor = torch.zeros(1, self.image_size, self.image_size)
        if post_path.exists():
            sar_image = self._load_image(post_path, mode='L')
            sar_image = sar_image.resize((self.image_size, self.image_size), Image.LANCZOS)
            sar_tensor = torch.from_numpy(np.array(sar_image)).unsqueeze(0).float() / 255.0
        else:
            self.logger.warning(f"SAR image not found: {post_path}, using zeros")

        # Load building damage mask
        mask_tensor = torch.zeros(1, self.image_size, self.image_size)
        if self.return_mask and mask_path and mask_path.exists():
            mask_image = self._load_image(mask_path, mode='L')
            mask_image = mask_image.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_array = np.array(mask_image)

            # Normalize mask to 0-3 range
            mask_normalized = self._normalize_mask(mask_array)
            mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).float()

        # Get disaster metadata
        disaster_type = torch.tensor(meta_entry.get('disaster_type', 0), dtype=torch.long)
        severity = torch.tensor(meta_entry.get('severity', 0.5), dtype=torch.float32)

        item = {
            'image_id': image_id,
            'rgb_image': rgb_tensor,
            'sar_image': sar_tensor,
            'building_mask': mask_tensor,
            'disaster_type': disaster_type,
            'disaster_severity': severity
        }

        return item


    def _find_file(self, directory: Path, filename_stem: str, extensions: list) -> Optional[Path]:
        """查找文件，支持多种扩展名和大小写"""
        if not directory.exists():
            return None

        for ext in extensions:
            for case_ext in (ext, ext.upper()):
                file_path = directory / f"{filename_stem}.{case_ext}"
                if file_path.exists():
                    return file_path
        return None

    def _load_image(self, path: Path, mode: str = 'RGB'):
        """加载图像，支持TIF格式"""
        path = Path(path)

        if path.suffix.lower() in ('.tif', '.tiff'):
            # TIF格式
            img_array = tifffile.imread(str(path))
            if mode == 'L' and img_array.ndim == 3:
                img_array = np.mean(img_array, axis=2)
            elif mode == 'RGB' and img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=2)
            img = Image.fromarray(img_array.astype(np.uint8))
        else:
            # JPG/PNG格式
            img = Image.open(path).convert(mode)

        return img

    def _normalize_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """标准化mask到0-3范围"""
        mask_unique = np.unique(mask_array)

        # 如果已经是0-3范围
        if mask_array.max() <= 3:
            return mask_array.astype(np.uint8)

        # 从[0, 255]映射到[0, 3]
        # 0-64 -> 0, 65-128 -> 1, 129-192 -> 2, 193-255 -> 3
        mask_normalized = np.round(mask_array / 85).astype(np.uint8)
        mask_normalized = np.clip(mask_normalized, 0, 3)

        return mask_normalized


class MetricsTracker:
    """Track and log training metrics"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_mean(self, key: str) -> float:
        """Get mean value for a metric"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(self.metrics[key])

    def reset(self):
        """Reset all metrics"""
        self.metrics = {}

    def log_epoch(self, epoch: int, lr: float = None):
        """Log metrics for an epoch"""
        log_str = f"Epoch {epoch}: "
        for key, values in self.metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                log_str += f"{key}={mean_val:.6f}, "

        if lr is not None:
            log_str += f"lr={lr:.6e}"

        self.logger.info(log_str)

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary of means"""
        return {key: self.get_mean(key) for key in self.metrics.keys()}


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    save_dir: str = "./checkpoints",
    is_best: bool = False,
    logger: logging.Logger = None
) -> str:
    """
    Save training checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        save_dir: Directory to save to
        is_best: Whether this is the best model
        logger: Logger instance

    Returns:
        Path to saved checkpoint
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }

    # Regular checkpoint
    ckpt_path = Path(save_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, ckpt_path)

    # Best model
    if is_best:
        best_path = Path(save_dir) / "best_model.pt"
        torch.save(checkpoint, best_path)
        if logger:
            logger.info(f"Saved best model to {best_path}")

    if logger:
        logger.info(f"Saved checkpoint to {ckpt_path}")

    return str(ckpt_path)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: str,
    logger: logging.Logger = None
) -> int:
    """
    Load training checkpoint

    Args:
        model: Model to load into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
        logger: Logger instance

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    if logger:
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")

    return epoch


def log_to_wandb(metrics: Dict[str, float], step: int = None):
    """Log metrics to Weights & Biases"""
    if not HAS_WANDB or wandb.run is None:
        return

    wandb.log(metrics, step=step)


# Disaster type utilities
def get_disaster_distribution(num_samples: int, distribution: str = "balanced") -> np.ndarray:
    """
    Get disaster type distribution for sampling

    Args:
        num_samples: Number of samples to generate distribution for
        distribution: "balanced" or "natural" (natural=more minor/major damage as per NeDS)

    Returns:
        Array of disaster types
    """
    if distribution == "balanced":
        # Uniform distribution over 5 disaster types
        return np.random.randint(0, 5, num_samples)
    elif distribution == "natural":
        # Match real-world distribution (biased towards certain types)
        # This could be customized based on actual data distribution
        return np.random.choice(
            [0, 1, 2, 3, 4],
            size=num_samples,
            p=[0.1, 0.3, 0.2, 0.2, 0.2]
        )
    else:
        return np.random.randint(0, 5, num_samples)


def get_severity_distribution(num_samples: int, distribution: str = "natural") -> np.ndarray:
    """
    Get disaster severity distribution for sampling

    Args:
        num_samples: Number of samples
        distribution: "uniform" or "natural" (natural=concentrated around 0.4 as per NeDS)

    Returns:
        Array of severity values in [0, 1]
    """
    if distribution == "uniform":
        return np.random.uniform(0, 1, num_samples)
    elif distribution == "natural":
        # NeDS uses [0.1, 0.4, 0.4, 0.1] probability distribution
        # Simulate this with a mixture of distributions
        severities = np.random.beta(4, 4, num_samples)  # Concentrated around 0.5
        severities = np.clip(severities * 0.8 + 0.1, 0, 1)  # Shift to [0.1, 0.9]
        return severities
    else:
        return np.random.uniform(0, 1, num_samples)

