import torch
from torch import nn
from diffusers.models.controlnet import ControlNetModel

class DisasterControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block_out_channels = kwargs['block_out_channels']
        
        # 删除原有condition embedding
        del self.controlnet_cond_embedding
        
        # 1. RGB图像处理器 
        self.rgb_processor = nn.Conv2d(3, block_out_channels[0], 3, padding=1)
        
        # 2. Mask处理器
        self.mask_processor = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            conditioning_channels=1,  # 单通道mask
            block_out_channels=(16, 32, 96, 128)
        )
        
        # 3. 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * block_out_channels[0], block_out_channels[0], 1),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], block_out_channels[0], 3, padding=1)
        )
        
        # 4. 灾害类型编码器 
        disaster_vec_dim = kwargs.get("disaster_vec_dim", 8)
        self.disaster_projection = nn.Sequential(
            nn.Linear(disaster_vec_dim, block_out_channels[0] * 4),
            nn.SiLU(),
            nn.Linear(block_out_channels[0] * 4, block_out_channels[0] * 4)
        )