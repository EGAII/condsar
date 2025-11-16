"""
Enhanced CONDSAR Model - Disaster-aware SAR Image Generation ControlNet
Based on NeDS architecture for RGB-to-SAR image generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

# Import MaskEncoder from training_utils
try:
    from .training_utils import MaskEncoder
except ImportError:
    from training_utils import MaskEncoder

# Import ControlNet with fallback
try:
    from diffusers.models.controlnets.controlnet import ControlNetModel, ControlNetConditioningEmbedding
except Exception:
    try:
        from diffusers.models.controlnet import ControlNetModel, ControlNetConditioningEmbedding
    except Exception:
        try:
            from diffusers.models.controlnets.controlnet import ControlNetModel
            ControlNetConditioningEmbedding = None
        except Exception:
            from diffusers.models.controlnet import ControlNetModel
            ControlNetConditioningEmbedding = None

# Fallback implementation for ControlNetConditioningEmbedding
if ControlNetConditioningEmbedding is None:
    class ControlNetConditioningEmbedding(nn.Module):
        """Simple fallback implementation for conditioning embedding."""
        def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 1,
            block_out_channels: Tuple[int, ...] = (16, 32, 96, 128)
        ):
            super().__init__()
            self.project = nn.Sequential(
                nn.Conv2d(conditioning_channels, conditioning_embedding_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
            )

        def forward(self, x):
            return self.project(x)


logger = logging.getLogger(__name__)


class DisasterTypeEmbedding(nn.Module):
    """
    Learnable disaster type embeddings (corresponds to NeDS learnable query embeddings)

    Args:
        num_disaster_types: Number of disaster types (default: 5 for Volcano, Earthquake, Wildfire, Storm, Flood)
        embedding_dim: Dimension of disaster type embeddings
    """
    def __init__(self, num_disaster_types: int = 5, embedding_dim: int = 128):
        super().__init__()
        self.num_disaster_types = num_disaster_types
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_disaster_types, embedding_dim)
        logger.info(f"Initialized DisasterTypeEmbedding: {num_disaster_types} types, dim={embedding_dim}")

    def forward(self, disaster_type_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disaster_type_idx: Tensor of shape (B,) with values in [0, num_disaster_types)

        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        return self.embedding(disaster_type_idx)


class DisasterSeverityEmbedding(nn.Module):
    """
    Learnable disaster severity/intensity embeddings
    Maps continuous severity values [0, 1] to learnable embeddings

    Args:
        num_severity_levels: Number of discrete severity levels
        embedding_dim: Dimension of severity embeddings
    """
    def __init__(self, num_severity_levels: int = 4, embedding_dim: int = 128):
        super().__init__()
        self.num_severity_levels = num_severity_levels
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_severity_levels, embedding_dim)
        logger.info(f"Initialized DisasterSeverityEmbedding: {num_severity_levels} levels, dim={embedding_dim}")

    def forward(self, severity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            severity: Tensor of shape (B,) with continuous values in [0, 1] or (B, num_severity_levels) one-hot

        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        if severity.dim() == 1:
            # Convert continuous [0,1] to discrete level
            severity_idx = (severity * (self.num_severity_levels - 1)).long()
            severity_idx = torch.clamp(severity_idx, 0, self.num_severity_levels - 1)
        else:
            # One-hot encoding: convert to index
            severity_idx = severity.argmax(dim=1)

        return self.embedding(severity_idx)


class EnhancedDisasterControlNet(ControlNetModel):
    """
    Enhanced ControlNet for Disaster-Aware SAR Image Generation

    Conditions:
    1. RGB pre-disaster optical image
    2. Building/damage mask
    3. Disaster type (learnable embedding)
    4. Disaster severity/intensity

    Args:
        num_disaster_types: Number of disaster types
        disaster_embedding_dim: Dimension for disaster type embeddings
        severity_embedding_dim: Dimension for severity embeddings
        **kwargs: Arguments passed to ControlNetModel
    """

    def __init__(
        self,
        num_disaster_types: int = 5,
        disaster_embedding_dim: int = 128,
        num_severity_levels: int = 4,
        severity_embedding_dim: int = 128,
        **kwargs
    ):
        # Extract before passing to super().__init__
        self.num_disaster_types = num_disaster_types
        self.disaster_embedding_dim = disaster_embedding_dim
        self.num_severity_levels = num_severity_levels
        self.severity_embedding_dim = severity_embedding_dim

        super().__init__(**kwargs)

        # Get block_out_channels from parent or kwargs
        block_out_channels = kwargs.get("block_out_channels", None)
        if block_out_channels is None:
            block_out_channels = getattr(self, "block_out_channels", (320, 640, 1280, 1280))

        self.block_out_channels = block_out_channels
        base_channels = block_out_channels[0]

        logger.info(f"Initializing EnhancedDisasterControlNet with block_out_channels={block_out_channels}")

        # ========== Condition Encoders ==========

        # 1. RGB Processor (pre-disaster optical image)
        self.rgb_processor = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, padding=3, stride=1),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        )

        # 2. Mask Processor (building/damage mask) - 可训练的MaskEncoder
        self.mask_encoder = MaskEncoder(
            input_channels=1,
            embedding_dim=base_channels,
            num_damage_levels=4,
            hidden_channels=64
        )

        # Mask空间特征处理

        # 3. Disaster Type Embedding
        self.disaster_type_embedding = DisasterTypeEmbedding(
            num_disaster_types=num_disaster_types,
            embedding_dim=disaster_embedding_dim
        )

        # 4. Disaster Severity Embedding
        self.severity_embedding = DisasterSeverityEmbedding(
            num_severity_levels=num_severity_levels,
            embedding_dim=severity_embedding_dim
        )

        # ========== Disaster Info Fusion ==========

        # Fuse disaster type and severity embeddings
        total_disaster_dim = disaster_embedding_dim + severity_embedding_dim
        self.disaster_fusion = nn.Sequential(
            nn.Linear(total_disaster_dim, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels)
        )

        # ========== Feature Fusion ==========

        # Fuse RGB + Mask spatial features
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(2 * base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels),
            nn.SiLU()
        )

        # Fuse spatial + disaster features
        self.multi_modal_fusion = nn.Sequential(
            nn.Conv2d(2 * base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels),
        )

        # Replace controlnet_cond_embedding with Identity to avoid double processing
        self.controlnet_cond_embedding = nn.Identity()

        # Encoder hidden states projection (for text embeddings if used)
        self.encoder_proj = None

        logger.info("EnhancedDisasterControlNet initialized successfully")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        rgb_image: Optional[torch.Tensor] = None,
        building_mask: Optional[torch.Tensor] = None,
        disaster_type: Optional[torch.Tensor] = None,
        disaster_severity: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for EnhancedDisasterControlNet

        Args:
            sample: Latent representation from diffusion (B, 4, H//8, W//8)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings (optional)
            rgb_image: Pre-disaster RGB image (B, 3, H, W)
            building_mask: Building/damage mask (B, 1, H, W)
            disaster_type: Disaster type indices (B,) with values in [0, num_disaster_types)
            disaster_severity: Disaster severity (B,) with values in [0, 1] or one-hot
            **kwargs: Additional arguments passed to parent

        Returns:
            Control embeddings to be passed to UNet
        """

        # ========== Process Spatial Conditions ==========
        rgb_feat = None
        mask_feat = None

        if rgb_image is not None:
            try:
                rgb_feat = self.rgb_processor(rgb_image)  # (B, C, H, W)
                logger.debug(f"RGB features shape: {rgb_feat.shape}")
            except Exception as e:
                logger.warning(f"RGB processing failed: {e}")
                rgb_feat = None

        if building_mask is not None:
            try:
                mask_feat = self.mask_processor(building_mask)  # (B, C, H, W)
                logger.debug(f"Mask features shape: {mask_feat.shape}")
            except Exception as e:
                logger.warning(f"Mask processing failed: {e}")
                mask_feat = None

        # Fuse RGB and Mask
        spatial_feat = None
        if rgb_feat is not None and mask_feat is not None:
            # Ensure spatial dimensions match
            if rgb_feat.shape[2:] != mask_feat.shape[2:]:
                mask_feat = F.adaptive_avg_pool2d(mask_feat, rgb_feat.shape[2:])
            spatial_feat = self.spatial_fusion(torch.cat([rgb_feat, mask_feat], dim=1))
            logger.debug(f"Fused spatial features shape: {spatial_feat.shape}")
        elif rgb_feat is not None:
            spatial_feat = rgb_feat
        elif mask_feat is not None:
            spatial_feat = mask_feat

        # ========== Process Disaster Conditions ==========
        disaster_feat = None
        if disaster_type is not None or disaster_severity is not None:
            # Get disaster type embedding
            if disaster_type is not None:
                disaster_type_emb = self.disaster_type_embedding(disaster_type)  # (B, dim1)
            else:
                disaster_type_emb = torch.zeros(
                    sample.size(0), self.disaster_embedding_dim,
                    device=sample.device, dtype=sample.dtype
                )

            # Get severity embedding
            if disaster_severity is not None:
                severity_emb = self.severity_embedding(disaster_severity)  # (B, dim2)
            else:
                severity_emb = torch.zeros(
                    sample.size(0), self.severity_embedding_dim,
                    device=sample.device, dtype=sample.dtype
                )

            # Fuse disaster type and severity
            combined_disaster = torch.cat([disaster_type_emb, severity_emb], dim=1)  # (B, dim1+dim2)
            disaster_proj = self.disaster_fusion(combined_disaster)  # (B, C)
            disaster_feat = disaster_proj.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

            # Expand to match sample spatial dimensions
            target_h, target_w = sample.size(2), sample.size(3)
            disaster_feat = disaster_feat.expand(-1, -1, target_h, target_w)  # (B, C, H, W)
            logger.debug(f"Disaster features shape: {disaster_feat.shape}")

        # ========== Multi-Modal Feature Fusion ==========
        fused = None
        if spatial_feat is not None and disaster_feat is not None:
            # Ensure dimensions match
            if spatial_feat.size(1) != disaster_feat.size(1):
                if spatial_feat.size(1) > disaster_feat.size(1):
                    disaster_feat = F.pad(
                        disaster_feat,
                        (0, 0, 0, 0, 0, spatial_feat.size(1) - disaster_feat.size(1))
                    )
                else:
                    spatial_feat = F.pad(
                        spatial_feat,
                        (0, 0, 0, 0, 0, disaster_feat.size(1) - spatial_feat.size(1))
                    )

            # Fuse spatial and disaster features
            fused = self.multi_modal_fusion(torch.cat([spatial_feat, disaster_feat], dim=1))
            logger.debug(f"Multi-modal fused features shape: {fused.shape}")
        elif spatial_feat is not None:
            fused = spatial_feat
        elif disaster_feat is not None:
            fused = disaster_feat

        # ========== Prepare Control Conditioning ==========
        if fused is not None:
            # Ensure fused spatial dimensions match sample
            if fused.dim() == 4 and sample.dim() == 4:
                target_h, target_w = sample.size(2), sample.size(3)
                if fused.size(2) != target_h or fused.size(3) != target_w:
                    fused = F.interpolate(
                        fused, size=(target_h, target_w),
                        mode="bilinear", align_corners=False
                    )

            # Ensure channel dimensions match
            try:
                expected_channels = self.conv_in.out_channels
            except Exception:
                expected_channels = fused.size(1)

            if fused.size(1) != expected_channels:
                if fused.size(1) > expected_channels:
                    fused = fused[:, :expected_channels, :, :]
                else:
                    fused = F.pad(fused, (0, 0, 0, 0, 0, expected_channels - fused.size(1)))

            kwargs['controlnet_cond'] = fused
            logger.debug(f"Final control conditioning shape: {fused.shape}")

        # ========== Handle Encoder Hidden States ==========
        if encoder_hidden_states is not None and encoder_hidden_states.dim() == 3:
            src_dim = encoder_hidden_states.size(-1)
            target_dim = None

            # Try to find target dimension from attention layers
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear) and "to_k" in name:
                    target_dim = m.in_features
                    break

            # Fallback strategies
            if target_dim is None:
                cfg = getattr(self, "config", None)
                if cfg is not None and hasattr(cfg, "cross_attention_dim"):
                    target_dim = cfg.cross_attention_dim
                elif hasattr(self, "block_out_channels"):
                    target_dim = self.block_out_channels[-1]

            # Apply projection if needed
            if target_dim is not None and src_dim != target_dim:
                need_new = True
                if getattr(self, "encoder_proj", None) is not None:
                    try:
                        if (self.encoder_proj.in_features == src_dim and
                            self.encoder_proj.out_features == target_dim):
                            need_new = False
                    except Exception:
                        need_new = True

                if need_new:
                    proj = nn.Linear(src_dim, target_dim)
                    device = sample.device if sample is not None else encoder_hidden_states.device
                    dtype = encoder_hidden_states.dtype
                    proj = proj.to(device=device, dtype=dtype)
                    self.encoder_proj = proj

                encoder_hidden_states = self.encoder_proj(encoder_hidden_states)

        return super().forward(sample, timestep, encoder_hidden_states, *args, **kwargs)


def create_enhanced_controlnet(
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base",
    num_disaster_types: int = 5,
    disaster_embedding_dim: int = 128,
    num_severity_levels: int = 4,
    severity_embedding_dim: int = 128,
    **kwargs
) -> EnhancedDisasterControlNet:
    """
    Create an EnhancedDisasterControlNet from a pretrained model

    Args:
        pretrained_model_name_or_path: Path or HuggingFace model ID
        num_disaster_types: Number of disaster types
        disaster_embedding_dim: Dimension for disaster type embeddings
        num_severity_levels: Number of severity levels
        severity_embedding_dim: Dimension for severity embeddings
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        EnhancedDisasterControlNet model
    """
    from diffusers import ControlNetModel

    # Load base ControlNet
    base_controlnet = ControlNetModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=kwargs.get("torch_dtype", torch.float32),
        **{k: v for k, v in kwargs.items() if k != "torch_dtype"}
    )

    # Convert to EnhancedDisasterControlNet
    model = EnhancedDisasterControlNet(
        in_channels=base_controlnet.config.in_channels,
        down_block_types=base_controlnet.config.down_block_types,
        block_out_channels=base_controlnet.config.block_out_channels,
        layers_per_block=base_controlnet.config.layers_per_block,
        cross_attention_dim=base_controlnet.config.cross_attention_dim,
        num_disaster_types=num_disaster_types,
        disaster_embedding_dim=disaster_embedding_dim,
        num_severity_levels=num_severity_levels,
        severity_embedding_dim=severity_embedding_dim,
    )

    # Copy weights from base model
    try:
        model.load_state_dict(base_controlnet.state_dict(), strict=False)
        logger.info("Successfully loaded base ControlNet weights")
    except Exception as e:
        logger.warning(f"Could not load all weights from base model: {e}")

    return model


class SARVAEDecoder(nn.Module):
    """
    SAR图像的VAE解码器 (可训练)
    将潜在表示解码为SAR图像
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        output_channels: int = 1,
        hidden_channels: int = 128
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        # 解码器网络
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                latent_channels, hidden_channels * 4,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            # 256x256 -> 512x512
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            # 输出层
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

        logger.info(f"Initialized SARVAEDecoder: latent={latent_channels}x{latent_size}x{latent_size}, output={output_channels}x512x512")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            latent: (B, 4, 64, 64) VAE latent

        Returns:
            sar_image: (B, 1, 512, 512)
        """
        sar_image = self.decoder(latent)
        return sar_image



