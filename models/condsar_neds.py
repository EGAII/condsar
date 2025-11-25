"""
CONDSAR NeDS Model - Disaster-aware SAR Image Generation ControlNet
Refactored based on NeDS architecture for RGB-to-SAR generation

Architecture:
- Input: RGB pre-disaster optical image (via frozen VAE encoder)
- Condition: Building/damage mask (via ControlNet embedding)
- Embeddings: Disaster type + severity (fused into time embedding)
- Output: SAR post-disaster image

Based on the paper: NeDS - Neural Disaster Simulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

# Import ControlNet base classes
try:
    from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
except ImportError:
    from diffusers.models.controlnets.controlnet import ControlNetModel, ControlNetOutput
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

logger = logging.getLogger(__name__)


def zero_module(module):
    """Initialize module parameters to zero"""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Mask Conditioning Embedding (similar to NeDS mask encoder)
    Encodes building/damage mask into feature space

    Mask values: 0=background, 1=intact, 2=damaged, 3=destroyed
    """
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1
        )

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        return embedding


class CONDSARNeDS(ControlNetModel):
    """
    CONDSAR NeDS Model - ControlNet-based RGB-to-SAR generation

    Similar to NeDS architecture but adapted for our use case:
    - pre_latents: RGB pre-disaster image encoded by frozen VAE
    - mask: Building/damage mask (0-3 values)
    - disaster_type: Disaster category (Volcano/Earthquake/Wildfire/Flood)
    - disaster_severity: Disaster intensity (0.0-1.0)

    Training follows NeDS Stage A:
    1. Encode pre-event RGB via frozen VAE -> pre_latents
    2. Encode mask via ControlNetConditioningEmbedding
    3. Add noise to pre_latents at timestep t
    4. Predict noise using ControlNet + disaster embeddings
    5. Compute diffusion loss
    """

    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        num_disaster_types = kwargs.pop('num_disaster_types', 5)
        conditioning_channels = kwargs.pop('conditioning_channels', 1)

        super().__init__(*args, **kwargs)

        block_out_channels = kwargs.get('block_out_channels', (320, 640, 1280, 1280))
        base_channels = block_out_channels[0]
        time_embed_dim = base_channels * 4

        # Delete the default controlnet_cond_embedding
        del self.controlnet_cond_embedding

        # ========== RGB Pre-Event Image Processing ==========
        # Project VAE latents to control features
        self.conv_latent = nn.Conv2d(
            4,  # VAE latent channels
            base_channels,
            kernel_size=3,
            padding=1
        )

        # ========== Mask Embedding ==========
        # Encode building/damage mask
        self.mask_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=base_channels,
            block_out_channels=(16, 32, 96, 256),
            conditioning_channels=conditioning_channels,
        )

        # ========== Condition Fusion ==========
        # Fuse RGB latent features + mask features
        self.conv_cond_fusion = nn.Sequential(
            nn.Conv2d(2 * base_channels, base_channels, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
        )

        # ========== Disaster Embeddings ==========
        # Disaster type embedding (learnable query embeddings like NeDS)
        self.disaster_type_embedding = nn.Embedding(num_disaster_types, time_embed_dim)

        # Disaster severity embedding (4 levels: none, minor, major, destroyed)
        self.severity_embedding = nn.Embedding(4, time_embed_dim)

        # Fuse disaster type and severity into time embedding space
        self.disaster_fusion = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        logger.info(f"Initialized CONDSARNeDS with {num_disaster_types} disaster types")

    def pre_conditioning(
        self,
        sample: torch.Tensor,
        pre_latents: torch.Tensor,
        mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process conditioning inputs (similar to NeDS pre_conditioning)

        Args:
            sample: Noisy latent (B, 4, 64, 64)
            pre_latents: Pre-event RGB latent from VAE encoder (B, 4, 64, 64)
            mask: Building/damage mask (B, 1, 512, 512) or (B, 1, 64, 64)
            encoder_hidden_states: Text embeddings (not used, kept for compatibility)

        Returns:
            Tuple of (processed_sample, control_cond, encoder_hidden_states)
        """
        # Process input sample
        sample = self.conv_in(sample)

        # Process pre-event RGB latent
        rgb_features = self.conv_latent(pre_latents)

        # Process mask
        # Resize mask if needed (should be same size as latent)
        if mask.shape[-2:] != pre_latents.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=pre_latents.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        mask_features = self.mask_embedding(mask)

        # Fuse RGB features + mask features
        controlnet_cond = torch.cat([rgb_features, mask_features], dim=1)
        controlnet_cond = self.conv_cond_fusion(controlnet_cond)

        # Add conditioning to sample
        sample = sample + controlnet_cond

        return sample, controlnet_cond, encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        pre_latents: torch.Tensor,
        mask: torch.Tensor,
        disaster_type: torch.Tensor,
        disaster_severity: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        Forward pass following NeDS architecture

        Args:
            sample: Noisy latent (B, 4, 64, 64)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings (optional, can be dummy)
            pre_latents: Pre-event RGB encoded by VAE (B, 4, 64, 64)
            mask: Building/damage mask (B, 1, H, W)
            disaster_type: Disaster type indices (B,)
            disaster_severity: Disaster severity level 0-3 (B,)
            conditioning_scale: Scaling factor for control signals
            return_dict: Whether to return ControlNetOutput

        Returns:
            ControlNetOutput with down_block_res_samples and mid_block_res_sample
        """
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # ========== Process Timestep ==========
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # ========== Add Disaster Embeddings to Time Embedding ==========
        # Get disaster type embedding
        disaster_type_emb = self.disaster_type_embedding(disaster_type.long())

        # Convert continuous severity [0, 1] to discrete level [0, 3]
        if disaster_severity.dtype == torch.float32 or disaster_severity.dtype == torch.float64:
            severity_idx = (disaster_severity * 3).long()
            severity_idx = torch.clamp(severity_idx, 0, 3)
        else:
            severity_idx = disaster_severity.long()

        severity_emb = self.severity_embedding(severity_idx)

        # Fuse disaster embeddings
        disaster_emb = torch.cat([disaster_type_emb, severity_emb], dim=1)
        disaster_emb = self.disaster_fusion(disaster_emb)

        # Add to time embedding (like NeDS adds disaster info to time embedding)
        emb = emb + disaster_emb

        aug_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError("text_embeds required for addition_embed_type='text_time'")
                text_embeds = added_cond_kwargs.get("text_embeds")
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # ========== Pre-conditioning (process control inputs) ==========
        sample, controlnet_cond, encoder_hidden_states = self.pre_conditioning(
            sample,
            pre_latents,
            mask,
            encoder_hidden_states
        )

        # ========== Down blocks ==========
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # ========== Mid block ==========
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # ========== ControlNet blocks ==========
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # ========== Scaling ==========
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample
        )


class SARVAEDecoder(nn.Module):
    """
    SAR VAE Decoder - Converts latent back to SAR image
    Trainable decoder for single-channel SAR output
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

        # Decoder: 64x64 -> 512x512
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                latent_channels, hidden_channels * 4,
                kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(32, hidden_channels * 4),
            nn.SiLU(),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2,
                kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(32, hidden_channels * 2),
            nn.SiLU(),

            # 256x256 -> 512x512
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(),

            # Final conv
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

        logger.info(f"Initialized SARVAEDecoder: {latent_channels}x{latent_size}x{latent_size} -> {output_channels}x512x512")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, 4, 64, 64) VAE latent

        Returns:
            sar_image: (B, 1, 512, 512) SAR image
        """
        return self.decoder(latent)


def create_condsar_neds_from_pretrained(
    pretrained_model_name: str = "stabilityai/stable-diffusion-2-1-base",
    num_disaster_types: int = 5,
    conditioning_channels: int = 1,
    torch_dtype: torch.dtype = torch.float32,
) -> CONDSARNeDS:
    """
    Create CONDSAR NeDS model from pretrained Stable Diffusion ControlNet

    Args:
        pretrained_model_name: Base model name
        num_disaster_types: Number of disaster types
        conditioning_channels: Number of mask channels (default: 1)
        torch_dtype: Model dtype

    Returns:
        CONDSARNeDS model
    """
    from diffusers import ControlNetModel

    # Try to load pretrained ControlNet or create from scratch
    try:
        base_controlnet = ControlNetModel.from_pretrained(
            pretrained_model_name,
            subfolder="controlnet",
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loaded pretrained ControlNet from {pretrained_model_name}")
    except Exception as e:
        logger.warning(f"Could not load pretrained ControlNet: {e}")
        logger.info("Creating ControlNet from scratch with SD2.1 config")
        # Create with SD2.1 default config
        base_controlnet = ControlNetModel(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            cross_attention_dim=1024,
            attention_head_dim=[5, 10, 20, 20],
        )

    # Create CONDSAR NeDS model
    model = CONDSARNeDS(
        in_channels=base_controlnet.config.in_channels,
        conditioning_channels=conditioning_channels,
        block_out_channels=base_controlnet.config.block_out_channels,
        layers_per_block=base_controlnet.config.layers_per_block,
        down_block_types=base_controlnet.config.down_block_types,
        cross_attention_dim=base_controlnet.config.cross_attention_dim,
        attention_head_dim=base_controlnet.config.attention_head_dim if hasattr(base_controlnet.config, 'attention_head_dim') else 8,
        num_disaster_types=num_disaster_types,
    )

    # Copy weights from base model (excluding our custom layers)
    try:
        model.load_state_dict(base_controlnet.state_dict(), strict=False)
        logger.info("Loaded base ControlNet weights (non-strict)")
    except Exception as e:
        logger.warning(f"Could not load base weights: {e}")

    return model

