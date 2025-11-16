# python
import torch
from torch import nn

# 优先从新路径导入；回退到旧路径；最后提供后备实现
try:
    # 推荐的新路径（避免 diffusers 的 deprecation 抛错）
    from diffusers.models.controlnets.controlnet import ControlNetModel, ControlNetConditioningEmbedding
except Exception:
    try:
        # 只导入 ControlNetModel（新路径首选）
        from diffusers.models.controlnets.controlnet import ControlNetModel
    except Exception:
        try:
            # 回退到旧路径（尽量避免，但保留以兼容老版本）
            from diffusers.models.controlnet import ControlNetModel, ControlNetConditioningEmbedding
        except Exception:
            try:
                from diffusers.models.controlnet import ControlNetModel
            except Exception:
                raise ImportError("diffusers.models.controlnets.controlnet.ControlNetModel 未找到。请安装兼容版本的 diffusers。")

            # 如果 ControlNetModel 可用，但 ControlNetConditioningEmbedding 不存在，使用后备
            ControlNetConditioningEmbedding = None
    else:
        # 新路径中没有 ControlNetConditioningEmbedding，尝试旧路径再后备
        try:
            from diffusers.models.controlnet import ControlNetConditioningEmbedding
        except Exception:
            ControlNetConditioningEmbedding = None

# 后备的 ControlNetConditioningEmbedding（仅在没有从 diffusers 成功导入时使用）
if "ControlNetConditioningEmbedding" not in globals() or ControlNetConditioningEmbedding is None:
    class ControlNetConditioningEmbedding(nn.Module):
        """
        简单后备实现：将 conditioning 投       影到指定的 embedding channels。
        仅作占位兼容使用。
        """
        def __init__(self, conditioning_embedding_channels, conditioning_channels=1, block_out_channels=(16, 32, 96, 128)):
            super().__init__()
            self.project = nn.Sequential(
                nn.Conv2d(conditioning_channels, conditioning_embedding_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(conditioning_embedding_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
            )

        def forward(self, x):
            return self.project(x)


# python
class DisasterControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        disaster_vec_dim = kwargs.pop("disaster_vec_dim", 8)
        block_out_channels = kwargs.get("block_out_channels", None)
        super().__init__(*args, **kwargs)
        if block_out_channels is None:
            block_out_channels = getattr(self, "block_out_channels", (16, 32, 96, 128))

        # RGB 与 mask 处理器
        self.rgb_processor = nn.Conv2d(3, block_out_channels[0], 3, padding=1)
        self.mask_processor = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            conditioning_channels=1,
            block_out_channels=(16, 32, 96, 128)
        )

        # 父类会调用 controlnet_cond_embedding，但我们在 forward 中传入已处理好的 embedding，
        # 所以这里使用 Identity 避免重复处理和通道不匹配
        self.controlnet_cond_embedding = nn.Identity()

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * block_out_channels[0], block_out_channels[0], 1),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], block_out_channels[0], 3, padding=1)
        )
        self.disaster_projection = nn.Sequential(
            nn.Linear(disaster_vec_dim, block_out_channels[0] * 4),
            nn.SiLU(),
            nn.Linear(block_out_channels[0] * 4, block_out_channels[0] * 4)
        )

        # 延迟创建 encoder 投影层（按需在 forward 中注册为模块 self.encoder_proj）
        self.encoder_proj = None

    def forward(self, sample, timestep, encoder_hidden_states, *args, rgb_image=None, mask=None, disaster_vec=None, **kwargs):
        import torch.nn.functional as F
        import torch

        fused = None
        rgb_emb = None
        mask_emb = None

        if rgb_image is not None:
            try:
                rgb_emb = self.rgb_processor(rgb_image)
            except Exception:
                rgb_emb = None

        if mask is not None:
            try:
                mask_emb = self.mask_processor(mask)
            except Exception:
                mask_emb = None

        if rgb_emb is not None and mask_emb is not None:
            if rgb_emb.shape[2:] != mask_emb.shape[2:]:
                # 将高分辨率下采样到低分辨率再融合（保持语义）
                rgb_emb = F.adaptive_avg_pool2d(rgb_emb, mask_emb.shape[2:])
            fused = self.feature_fusion(torch.cat([rgb_emb, mask_emb], dim=1))
        elif rgb_emb is not None:
            fused = rgb_emb
        elif mask_emb is not None:
            fused = mask_emb

        if disaster_vec is not None:
            proj = self.disaster_projection(disaster_vec)  # (B, C)
            proj = proj.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            target_h = None
            target_w = None
            if fused is not None and fused.dim() == 4:
                target_h, target_w = fused.size(2), fused.size(3)
            elif sample is not None and sample.dim() == 4:
                # 父类的 conv_in 不改变空间分辨率，因此使用原始 sample 的 H/W 作为目标
                target_h, target_w = sample.size(2), sample.size(3)
            if target_h is not None and target_w is not None:
                proj = proj.expand(-1, -1, target_h, target_w)
            if fused is not None:
                if proj.size(1) != fused.size(1):
                    c_target = fused.size(1)
                    c_proj = proj.size(1)
                    if c_proj > c_target:
                        proj = proj[:, :c_target, :, :]
                    else:
                        proj = torch.nn.functional.pad(proj, (0, 0, 0, 0, 0, c_target - c_proj))
                fused = fused + proj
            else:
                fused = proj

        if fused is not None:
            # 确保 fused 的空间尺寸与 parent 的 sample 相同（parent 在这之前仅对 sample 做 conv_in，保持 H/W）
            if sample is not None and fused.dim() == 4:
                target_h, target_w = sample.size(2), sample.size(3)
                if fused.size(2) != target_h or fused.size(3) != target_w:
                    fused = F.interpolate(fused, size=(target_h, target_w), mode="bilinear", align_corners=False)

            # 若通道数仍不匹配（极少见），做通道截断或填充以对齐
            try:
                sample_channels = self.conv_in(sample).size(1)
            except Exception:
                sample_channels = fused.size(1)
            if fused.size(1) != sample_channels:
                c_target = sample_channels
                c_fused = fused.size(1)
                if c_fused > c_target:
                    fused = fused[:, :c_target, :, :]
                else:
                    fused = torch.nn.functional.pad(fused, (0, 0, 0, 0, 0, c_target - c_fused))

            kwargs['controlnet_cond'] = fused

        # --- 这里处理 encoder_hidden_states 的维度不匹配问题 ---
        if encoder_hidden_states is not None and encoder_hidden_states.dim() == 3:
            src_dim = encoder_hidden_states.size(-1)
            # 尝试从子模块中找到 attention 的 to_k 层以获取期望维度
            target_dim = None
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear) and "to_k" in name:
                    target_dim = m.in_features
                    break
            # 退回策略：尝试 config.cross_attention_dim 或 block_out_channels 最后一个
            if target_dim is None:
                cfg = getattr(self, "config", None)
                if cfg is not None and getattr(cfg, "cross_attention_dim", None):
                    target_dim = cfg.cross_attention_dim
                else:
                    block_out = getattr(self, "block_out_channels", None)
                    if block_out:
                        target_dim = block_out[-1]
            # 如果仍未找到，就不做投影
            if target_dim is not None and src_dim != target_dim:
                # 仅当需要且尚未注册或维度不匹配时创建/替换投影层
                need_new = True
                if getattr(self, "encoder_proj", None) is not None:
                    try:
                        if (self.encoder_proj.in_features == src_dim and self.encoder_proj.out_features == target_dim):
                            need_new = False
                    except Exception:
                        need_new = True
                if need_new:
                    # 创建投影并移动到正确设备/dtype，然后注册到模块上
                    proj = nn.Linear(src_dim, target_dim)
                    # 将模块移到 sample 的 device 与 encoder_hidden_states 的 dtype（若 sample 可用）
                    device = sample.device if sample is not None else encoder_hidden_states.device
                    dtype = encoder_hidden_states.dtype
                    proj = proj.to(device=device, dtype=dtype)
                    self.encoder_proj = proj
                # 应用投影
                encoder_hidden_states = self.encoder_proj(encoder_hidden_states)

        return super().forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

