import torch
from ...models.condsar import DisasterControlNet

def validate_model_components():
    """验证模型各个组件的功能"""
    model = DisasterControlNet(
        in_channels=4,
        block_out_channels=(320, 640, 1280, 1280)
    )
    
    # 1. 测试RGB处理
    rgb = torch.randn(1, 3, 64, 64)
    rgb_feat = model.rgb_processor(rgb)
    print(f"RGB feature shape: {rgb_feat.shape}")
    
    # 2. 测试Mask处理
    mask = torch.randn(1, 1, 64, 64)
    mask_feat = model.mask_processor(mask)
    print(f"Mask feature shape: {mask_feat.shape}")
    
    # 3. 测试特征融合
    fused = model.feature_fusion(torch.cat([rgb_feat, mask_feat], dim=1))
    print(f"Fused feature shape: {fused.shape}")
    
    # 4. 测试灾害向量投影
    disaster_vec = torch.randn(1, 8)
    time_embed = model.disaster_projection(disaster_vec)
    print(f"Time embedding shape: {time_embed.shape}")

if __name__ == "__main__":
    validate_model_components()