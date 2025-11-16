# CONDSAR 完整使用指南


---

##  环境准备

### 验证环境

```bash
cd D:\condsar
python quick_verify.py
```

检查项:
-  PyTorch 版本
-  CUDA 可用
-  GPU 类型和显存
-  数据完整性

### 安装依赖

```bash
pip install -r requirements_condsar.txt
```

关键包:
- torch >= 2.0
- diffusers >= 0.25
- transformers >= 4.35
- wandb
- pyyaml

---

##  项目结构

```
D:\condsar\
├── config_training.yaml        配置文件 (集中管理参数)
├── load_config.py              配置管理工具
├── QUICKSTART.md               快速开始
├── ARCHITECTURE.md             模型架构
├── GUIDE.md                    完整指南
├── PROJECT_STRUCTURE.md        项目结构说明
│
├── models/                     模型实现
│   ├── enhanced_condsar.py     ControlNet 模型 ⭐
│   ├── training_stage_a.py     Stage A 训练
│   ├── training_utils.py       数据和工具
│   ├── weighted_sampler.py     加权采样
│   └── ...
│
├── scripts/                    执行脚本
│   ├── train.py               训练脚本 ⭐ (支持配置文件加载)
│   ├── inference.py           推理脚本
│   └── verify.py              验证脚本
│
├── data/                       数据目录
│   ├── metadata.json          元数据 (3155 条)
│   ├── pre/                   RGB 灾前
│   ├── post/                  SAR 灾后
│   └── mask/                  建筑掩码
│
└── outputs/                    输出结果
    ├── checkpoints/           模型检查点
    ├── logs/                  训练日志
    └── results/               推理结果
```

---

##  训练流程

### 配置文件加载

#### 方式 1: 仅使用配置文件

```bash
python scripts/train.py --config config_training.yaml
```

#### 方式 2: 配置文件 + 命令行参数

```bash
python scripts/train.py \
    --config config_training.yaml \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --num-epochs 50
```

**优先级**: 命令行参数 > 配置文件

#### 方式 3: 仅命令行参数

```bash
python scripts/train.py \
    --stage a \
    --batch-size 8 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --use-wandb
```

### 配置管理工具

```bash
# 显示配置
python load_config.py --config config_training.yaml --show-config

# 验证配置
python load_config.py --config config_training.yaml --validate

# 生成命令
python load_config.py --config config_training.yaml --generate-command

# 保存为 JSON
python load_config.py --config config_training.yaml --output-json config.json
```

### Stage A: 源域训练

```bash
# 快速测试 (5分钟)
python scripts/train.py --config config_training.yaml --batch-size 2 --num-epochs 2

# 标准训练 (推荐)
python scripts/train.py --config config_training.yaml

# 自定义参数
python scripts/train.py \
    --config config_training.yaml \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --num-epochs 100
```

**预期**:
- 时间: ~100分钟 (100 epochs, A100)
- 显存: ~11.6GB
- 输出: `outputs/checkpoints/best_model.pt`

### 性能参数

| 参数 | 默认值 | 推荐值 | 快速测试 |
|------|--------|--------|---------|
| batch_size | 4 | 8 | 2 |
| num_epochs | 100 | 100 | 2 |
| learning_rate | 1e-4 | 1e-4 | 1e-4 |
| weight_decay | 1e-5 | 1e-5 | 1e-5 |

---

##  数据管理

### 元数据结构

```json
{
  "sample_001": {
    "pre_event": "pre/sample_001.tif",
    "post_event": "post/sample_001.tif",
    "mask": "mask/sample_001.tif",
    "disaster_type": "Volcano",
    "disaster_id": 0
  }
}
```

### 灾害类型映射

```
0 = Volcano (火山)
1 = Earthquake (地震)
2 = Wildfire (野火)
3 = Flood (洪水)
```

### 掩码值含义

```
0 = Background (背景)
1 = Intact (完好)
2 = Light Damaged (轻度损伤)
3 = Destroyed (严重损伤/摧毁)
```

---

##  加权采样

### 启用方式

自动启用 (无需额外配置)

### 权重策略

在 `config_training.yaml` 中配置:

```yaml
weighted_sampler:
  enabled: true
  strategy: "inverse_frequency"  # 推荐
  temperature: 1.0               # 权重平滑因子
```

### 计算的权重

```
Volcano: 0.747 (欠采样)
Earthquake: 0.430 (欠采样)
Wildfire: 5.570 (过采样)
Flood: 6.357 (过采样)
```

**效果**: 所有类别基本等概率采样 

---

##  推理

### 单个样本推理

```bash
python scripts/inference.py \
    --mode single \
    --rgb-image ./data/pre/sample_001.tif \
    --building-mask ./data/mask/sample_001.tif \
    --checkpoint ./outputs/checkpoints/best_model.pt
```

### 批量推理

```bash
python scripts/inference.py \
    --mode batch \
    --rgb-dir ./data/pre \
    --mask-dir ./data/mask \
    --checkpoint ./outputs/checkpoints/best_model.pt \
    --batch-size 8
```

### 指定灾害条件

```bash
python scripts/inference.py \
    --mode batch \
    --rgb-dir ./data/pre \
    --mask-dir ./data/mask \
    --checkpoint ./outputs/checkpoints/best_model.pt \
    --disaster-type 1 \          # Earthquake
    --severity 0.8               # 80% 强度
```

---

##  监控和日志

### W&B Dashboard

```
https://wandb.ai/your-username/condsar
```

监控内容:
- 训练损失曲线
- 各灾害类型损失
- 学习率变化
- 特征图可视化
- 样本输出

### 本地日志

```bash
# 实时查看日志
Get-Content -Path ./outputs/logs/condsar_trainer_*.log -Wait

# 查看完整日志
Get-Content -Path ./outputs/logs/condsar_trainer_*.log

# 查看训练指标
Get-Content -Path ./outputs/metrics.json
```

---

##  检查点管理

### 保存位置

```
outputs/checkpoints/
├── best_model.pt              最优模型 (自动加载恢复)
├── checkpoint_epoch_010.pt    周期检查点
├── checkpoint_epoch_020.pt
└── ...
```

### 检查点内容

```python
{
    'epoch': 10,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'loss': 0.1234,
    'config': {...}
}
```

### 加载检查点

```python
checkpoint = torch.load('./outputs/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

##  常见问题

### Q1: 配置文件无法加载?

**检查**:
```bash
python -c "
import yaml
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(' Config loaded')
"
```

### Q2: CUDA 显存不足?

**解决**:
```bash
# 减小 batch_size
python scripts/train.py --config config_training.yaml --batch-size 2

# 或启用梯度检查点
# (需要在代码中启用)
```

### Q3: 如何验证数据?

**命令**:
```bash
python quick_verify.py --check-data

# 或
python verify_metadata.py
```

### Q4: W&B 连接失败?

**离线模式**:
```bash
python scripts/train.py --config config_training.yaml --wandb-offline
```

---

##  完整工作流

### 1. 环境验证

```bash
python quick_verify.py
```

### 2. 启动训练

```bash
python scripts/train.py --config config_training.yaml
```

### 3. 监控进度

```
W&B Dashboard: https://wandb.ai/your-username/condsar
本地日志: outputs/logs/
```

### 4. 等待完成

```
预计时间: 100-150 分钟 (100 epochs)
```

### 5. 推理生成

```bash
python scripts/inference.py --mode batch --batch-size 8
```

### 6. 查看结果

```
输出位置: outputs/results/
模型位置: outputs/checkpoints/best_model.pt
日志位置: outputs/logs/
```

---

## ️ 高级配置

### 自定义训练参数

在 `config_training.yaml` 中修改:

```yaml
training:
  stage_a:
    batch_size: 16              # 增大 batch size
    num_epochs: 50              # 减少 epochs
    learning_rate: 5e-5         # 降低学习率
    weight_decay: 1e-5
    gradient_accumulation_steps: 2
```

### 启用混合精度

```yaml
training:
  stage_a:
    use_mixed_precision: true   # FP16 加速
```

### 调整加权采样

```yaml
weighted_sampler:
  strategy: "sqrt_frequency"    # 或 "custom"
  temperature: 2.0              # 调整平衡度
```

---

##  项目结构说明

详见: `PROJECT_STRUCTURE.md`

---

##  获取帮助

| 问题类型 | 位置 |
|---------|------|
| 快速开始 | `QUICKSTART.md` |
| 模型架构 | `ARCHITECTURE.md` |
| 配置参数 | `config_training.yaml` |
| 项目结构 | `PROJECT_STRUCTURE.md` |

---

##  检查清单

启动前:
- [ ] 运行 `python quick_verify.py`
- [ ] 检查 `data/metadata.json` 存在
- [ ] 确认 CUDA 可用
- [ ] 检查磁盘空间 (>50GB 建议)

训练中:
- [ ] 监控 W&B Dashboard
- [ ] 定期检查日志
- [ ] 观察显存占用 (`nvidia-smi`)

训练后:
- [ ] 验证 `outputs/checkpoints/best_model.pt` 存在
- [ ] 检查训练日志是否完整
- [ ] 保存重要文件

---

**文档完成**:   
**版本**: 2.0  
**状态**: 生产就绪


