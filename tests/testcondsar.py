import torch
import pytest
from ..models.condsar import DisasterControlNet

class TestDisasterControlNet:
    @pytest.fixture
    def model(self):
        return DisasterControlNet(
            in_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            disaster_vec_dim=8
        )
    
    def test_model_initialization(self, model):
        """测试模型初始化"""
        assert isinstance(model, DisasterControlNet)
        assert model.rgb_processor is not None
        assert model.mask_processor is not None
        assert model.feature_fusion is not None
        assert model.disaster_projection is not None
    
    def test_forward_shapes(self, model):
        """测试前向传播的张量形状"""
        batch_size = 2
        # 创建测试输入
        sample = torch.randn(batch_size, 4, 64, 64)
        timestep = torch.tensor([1])
        rgb_image = torch.randn(batch_size, 3, 64, 64) 
        mask = torch.randn(batch_size, 1, 64, 64)
        disaster_vec = torch.randn(batch_size, 8)
        encoder_hidden_states = torch.randn(batch_size, 77, 768)
        
        # 前向传播
        output = model(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            rgb_image=rgb_image,
            mask=mask,
            disaster_vec=disaster_vec
        )
        
        # 验证输出形状
        assert len(output.down_block_res_samples) == 13  # ControlNet默认下采样块数
        assert output.mid_block_res_sample.shape == (batch_size, 1280, 8, 8)