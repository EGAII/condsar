# CONDSAR - ControlNet for Disaster SAR Generation

**版本**: 2.0 | **日期**: 2025-11-17 | **状态**: 生产就绪 ✅

---

## 📖 核心文档 (3 个)

**快速选择你的路径**:

| 我想... | 文档 | 时间 |
|---------|------|------|
| **快速启动** | [QUICKSTART.md](QUICKSTART.md) | 5分钟 ⚡ |
| **理解架构** | [ARCHITECTURE.md](ARCHITECTURE.md) | 15分钟 🧠 |
| **完整学习** | [GUIDE.md](GUIDE.md) | 1小时 📚 |

---

## ⚡ 3 分钟快速开始

```bash
# 1. 验证环境
python quick_verify.py

# 2. 启动训练
python scripts/train.py --config config_training.yaml

# 3. 监控进度
# → https://wandb.ai/your-username/condsar
```

---

## 🎯 项目概述

### 任务

将灾难前的光学RGB图像转换为灾难后的SAR图像，同时考虑：
- 建筑损伤掩码 (0-3: 背景/完好/轻损/重损)
- 灾害类型 (Volcano/Earthquake/Wildfire/Flood)
- 灾害强度 (0.0-1.0)

### 方法

基于 **ControlNet** + **Stable Diffusion 2.1**，使用加权采样处理数据不均衡

### 数据

```
总样本: 3155
├─ Volcano: 1056 (33.5%)
├─ Earthquake: 1833 (58.1%)  ← 最多
├─ Wildfire: 142 (4.5%)
└─ Flood: 124 (3.9%)         ← 最少

不均衡系数: 14.76:1 ✅ (自动处理)
```

---

## 🚀 主要特性

✅ **配置文件支持** - YAML/JSON 灵活配置  
✅ **加权采样** - 自动处理数据不均衡  
✅ **W&B 集成** - 实时监控训练过程  
✅ **多模态融合** - RGB + Mask + 灾害类型 + 强度  
✅ **生产就绪** - 经过验证的训练管道  
✅ **文档完整** - 快速/架构/详细三层文档  

---

## 📁 项目结构

```
D:\condsar\
├── 📖 文档 (3个核心)
│   ├── QUICKSTART.md          快速开始
│   ├── ARCHITECTURE.md        模型架构
│   ├── GUIDE.md              完整指南
│   └── PROJECT_STRUCTURE.md  项目结构
│
├── ⚙️ 配置
│   ├── config_training.yaml   训练配置
│   └── load_config.py         配置工具
│
├── 🧠 模型代码 (models/)
│   ├── enhanced_condsar.py    ControlNet
│   ├── training_stage_a/b/c   三阶段训练
│   ├── weighted_sampler.py    加权采样
│   └── training_utils.py      数据工具
│
├── 🚀 脚本 (scripts/)
│   ├── train.py              训练 ⭐ (支持配置文件)
│   ├── inference.py          推理
│   └── verify.py             验证
│
├── 📊 数据 (data/)
│   ├── metadata.json         3155条记录
│   ├── pre/                  RGB灾前
│   ├── post/                 SAR灾后
│   └── mask/                 建筑掩码
│
└── 📤 输出 (outputs/)
    ├── checkpoints/          模型
    ├── logs/                 日志
    └── results/              推理结果
```

---

## 🔧 使用方式

### 方式 1: 配置文件 (推荐)

```bash
# 直接使用配置文件
python scripts/train.py --config config_training.yaml

# 配置文件 + 命令行参数 (参数优先)
python scripts/train.py --config config_training.yaml --batch-size 16
```

### 方式 2: 命令行参数

```bash
python scripts/train.py \
    --stage a \
    --batch-size 8 \
    --num-epochs 100 \
    --use-wandb
```

### 配置验证

```bash
# 显示配置
python load_config.py --config config_training.yaml --show-config

# 验证配置
python load_config.py --config config_training.yaml --validate

# 生成命令
python load_config.py --config config_training.yaml --generate-command
```

---

## 📊 模型架构 (简要)

```
输入: RGB灾前 + Building Mask + Disaster Type + Severity
  ↓
处理: 4个条件 → Embedding 转换
  ↓
融合: 多模态特征融合
  ↓
输出: SAR灾后图像
```

**详见**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## ⏱️ 运行时间估计

| 操作 | 时间 | GPU |
|------|------|-----|
| 验证环境 | 30秒 | 任何 |
| 数据加载 | 2分钟 | 任何 |
| Stage A (100 epochs) | 100分钟 | A100 |

---

## 📝 配置文件 (`config_training.yaml`)

```yaml
# 核心参数
training:
  stage_a:
    batch_size: 8
    num_epochs: 100
    learning_rate: 1e-4

# 加权采样 (自动处理不均衡)
weighted_sampler:
  enabled: true
  strategy: "inverse_frequency"

# W&B 监控
wandb:
  enabled: true
  project: "condsar"
```

---

## 🎯 三阶段训练

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **A** | RGB + SAR + Mask + Type | best_model.pt | 源域训练 ✅ |
| **B** | RGB + Mask (仅Type) | 合成SAR | 目标域生成 (可选) |
| **C** | 真实 + 合成数据 | 微调模型 | 混合训练 (可选) |

---

## ✅ 检查清单

启动前:
- [ ] `python quick_verify.py` 通过
- [ ] `data/metadata.json` 存在
- [ ] CUDA 可用

启动后:
- [ ] 监控 W&B Dashboard
- [ ] 定期查看日志

---

## 🔗 快速链接

| 链接 | 说明 |
|------|------|
| [QUICKSTART.md](QUICKSTART.md) | 快速开始指南 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 详细架构分析 |
| [GUIDE.md](GUIDE.md) | 完整使用说明 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 项目结构 |
| [config_training.yaml](config_training.yaml) | 训练配置 |

---

## 🚀 立即开始

```bash
cd D:\condsar
python quick_verify.py
python scripts/train.py --config config_training.yaml
```

**预期**: 100分钟后得到最优模型 ✅

---

**更多详情请阅读对应的文档** → [QUICKSTART.md](QUICKSTART.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [GUIDE.md](GUIDE.md)


