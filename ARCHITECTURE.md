# CONDSAR 模型架构说明

---

##  模型整体架构

```
输入层:
├─ RGB Pre-disaster (B,3,512,512)
├─ Building Mask (B,1,512,512)          [0-3: 背景/完好/轻损/重损]
├─ Disaster Type (B,)                   [0-3: Volcano/Earthquake/Wildfire/Flood]
└─ Disaster Severity (B,)               [0.0-1.0: 灾害强度]

        ↓ [处理转换]

条件处理:
├─ RGB → VAE Encoder (冻结) → Latent (B,4,64,64) → 投影 → RGB Features (B,320,64,64)
├─ Mask → MaskEncoder (可训练) → Mask Features (B,320,64,64)
├─ Type → Embedding (可训练) → 投影+广播 → Type Features (B,320,64,64)
└─ Severity → 离散化 → Embedding (可训练) → 投影+广播 → Severity Features (B,320,64,64)

        ↓ [多模态融合]

融合层:
├─ RGB + Mask → Spatial Fusion → (B,320,64,64)
├─ Type + Severity → Disaster Fusion → (B,320,64,64)
└─ Spatial + Disaster → Multi-Modal Fusion → Control Embeddings (B,320,64,64)

        ↓ [生成]

输出: SAR 灾后图像 (B,1,512,512)
```

---

##  Condition → Embedding 转换

### 1. RGB 图像处理

```
RGB (B,3,512,512)
  ↓
VAE Encoder (冻结 ️)       [来自 Stable Diffusion 2.1]
  ↓
RGB Latent (B,4,64,64)
  ↓
投影层 (可训练 )            [3层卷积]
  ↓
RGB Features (B,320,64,64)
```

**关键点**:
- VAE Encoder 冻结 (不训练)
- 只有投影层可训练
- 将高维特征压缩到控制空间

### 2. Building Mask 处理

```
Mask (B,1,512,512) [值: 0,1,2,3]
  ↓
MaskEncoder (可训练 )       [5层卷积 + 跳连]
  ├─ Conv(1→64)
  ├─ Conv(64→128, stride=2)  [降采样]
  └─ Conv(128→320, stride=2)
  ↓
Mask Features (B,320,64,64)
```

**关键点**:
- 完全可训练
- 学习 Mask 的特征表示
- 帮助模型理解建筑损伤

### 3. Disaster Type 处理

```
Type (B,) [0=Volcano, 1=Earthquake, 2=Wildfire, 3=Flood]
  ↓
Embedding(4, 128) (可训练 )
  ├─ 每个灾害类型有 128 维向量
  └─ 向量在训练中学习
  ↓
Type Embedding (B,128)
  ↓
投影 + 广播 (1,1) → (64,64)
  ↓
Type Features (B,320,64,64)
```

**关键点**:
- 离散化 → 连续向量
- 模型学习灾害特征
- 实现条件控制

### 4. Disaster Severity 处理

```
Severity (B,) [0.0-1.0]
  ↓
离散化到 [0,1,2,3]           [分为4个强度级别]
  ↓
Embedding(4, 128) (可训练 )
  ↓
Severity Embedding (B,128)
  ↓
投影 + 广播
  ↓
Severity Features (B,320,64,64)
```

**关键点**:
- 连续值 → 离散化 → 嵌入
- 模型学习不同强度特征
- 强度影响生成的 SAR 信号

---

##  多模态融合

### 融合流程

```
Step 1: 空间融合
  RGB (B,320,64,64) + Mask (B,320,64,64)
    ↓ Concat
  (B,640,64,64)
    ↓ Conv 融合
  Spatial Features (B,320,64,64)

Step 2: 灾害特征准备
  Type (B,320,64,64) + Severity (B,320,64,64)
    ↓ Concat + 广播
  Disaster Features (B,640,64,64)

Step 3: 多模态融合
  Spatial + Disaster
    ↓ Concat
  (B,960,64,64)
    ↓ Conv 融合
  Control Embeddings (B,320,64,64) 
```

### 融合公式

```
S = Conv(Concat([RGB_feat, Mask_feat]))              # 空间融合
D = Broadcast(Concat([Type_emb, Severity_emb]))      # 灾害特征
C = Conv(Concat([S, D]))                             # 多模态融合
```

---

##  核心模块

### EnhancedDisasterControlNet

```python
class EnhancedDisasterControlNet(nn.Module):
    def __init__(self, ...):
        # 输入处理
        self.rgb_processor = RGBProcessor()
        self.mask_encoder = MaskEncoder()
        
        # 条件处理
        self.disaster_type_embedding = DisasterTypeEmbedding()
        self.disaster_severity_embedding = DisasterSeverityEmbedding()
        
        # 融合
        self.spatial_fusion = SpatialFusion()
        self.multi_modal_fusion = MultiModalFusion()
```

### 可训练组件 ()

| 组件 | 参数数 | 说明 |
|------|--------|------|
| RGB 投影 | 1M | VAE latent → 控制空间 |
| MaskEncoder | 2M | 建筑 Mask 特征 |
| Type Embedding | 650K | 灾害类型表示 |
| Severity Embedding | 650K | 灾害强度表示 |
| Spatial Fusion | 800K | RGB + Mask 融合 |
| Multi-Modal Fusion | 1.2M | 空间 + 灾害融合 |
| SARVAEDecoder | 2M | Latent → SAR 图像 |

**总参数**: ~8.5M (可训练)

### 冻结组件 (️)

| 组件 | 说明 |
|------|------|
| VAE Encoder | 来自 Stable Diffusion 2.1 |
| UNet | 来自 Stable Diffusion 2.1 |

---

##  数据维度变化

```
输入:
├─ RGB: (B=4, 3, 512, 512)
├─ Mask: (B=4, 1, 512, 512)
├─ Type: (B=4,) [标量]
└─ Severity: (B=4,) [标量]

处理:
├─ RGB latent: (4, 4, 64, 64)
├─ RGB features: (4, 320, 64, 64)
├─ Mask features: (4, 320, 64, 64)
├─ Type emb: (4, 128) → (4, 320, 1, 1) → (4, 320, 64, 64)
└─ Severity emb: (4, 128) → (4, 320, 1, 1) → (4, 320, 64, 64)

融合:
├─ Spatial: (4, 320, 64, 64)
├─ Disaster: (4, 320, 64, 64)
└─ Control: (4, 320, 64, 64) 

输出:
└─ SAR: (4, 1, 512, 512)
```

---

## ️ 加权采样 (数据不均衡处理)

### 问题

```
数据分布:
  Volcano: 1056 (33.5%)
  Earthquake: 1833 (58.1%) ← 最多
  Wildfire: 142 (4.5%)
  Flood: 124 (3.9%)    ← 最少

不均衡系数: 14.76:1
```

### 解决方案

```
权重公式: w_i = N_total / (num_classes × N_i)

计算:
  w_volcano = 3155 / (4 × 1056) = 0.747
  w_earthquake = 3155 / (4 × 1833) = 0.430
  w_wildfire = 3155 / (4 × 142) = 5.570
  w_flood = 3155 / (4 × 124) = 6.357

效果: 所有类别基本等概率采样 
```

---

##  训练数据流

```
Stage A (源域训练):

输入数据:
├─ RGB 灾前
├─ SAR 灾后
├─ Building Mask
└─ Disaster Type

        ↓

VAE 编码:
├─ RGB → RGB Latent
└─ SAR → SAR Latent

        ↓

添加扩散噪声:
└─ SAR Latent + Noise → Noisy SAR

        ↓

CONDSAR 处理:
└─ Conditions → Control Embeddings

        ↓

UNet 去噪:
├─ Input: Noisy SAR + Timestep
├─ Control: Control Embeddings
└─ Output: Predicted Noise

        ↓

损失计算:
└─ MSE Loss = ||Predicted Noise - True Noise||²

        ↓

优化:
├─ Backward Pass
├─ Optimizer Step
└─ Update Parameters
```

---

##  基础模型

```
Stable Diffusion 2.1 (stabilityai/stable-diffusion-2-1-base)
├─ VAE (Variational Autoencoder)
│   ├─ Encoder: Image → Latent (冻结 ️)
│   └─ Decoder: Latent → Image
├─ UNet (Denoising Model)
│   ├─ Multi-scale features
│   └─ Attention mechanisms
└─ Text Encoder (CLIP) [可选]
```

---

##  模型优化指标

| 指标 | 值 |
|------|-----|
| 显存占用 (A100) | ~11.6GB |
| 训练时间/epoch | ~1分钟 |
| 推理时间/样本 | ~2秒 |
| 可训练参数 | 8.5M |
| 总参数 | ~900M (含冻结) |

---

##  模型总结

```
输入: RGB灾前 + Mask + 灾害类型 + 强度
处理: 4个条件 → Embedding → 多模态融合 → 控制信号
输出: SAR灾后

特点:
 多模态条件控制
 自动加权采样处理不均衡
 高效的特征融合
 可追踪的 W&B 集成
```


