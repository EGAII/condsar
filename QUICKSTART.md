# CONDSAR 快速开始指南
---

##  3 分钟快速开始

### Step 1: 验证环境

```bash
cd D:\condsar
python quick_verify.py
```

**预期输出**:
```
 PyTorch version: 2.1.0+cu118
 CUDA available: True
 GPU: NVIDIA A100
 Data verified: 3155 samples
```

### Step 2: 启动训练

```bash
# 使用配置文件 (推荐)
python scripts/train.py --config config_training.yaml

# 或使用命令行参数
python scripts/train.py --stage a --batch-size 8 --num-epochs 100 --use-wandb
```

### Step 3: 监控训练

打开 W&B Dashboard: https://wandb.ai/your-username/condsar

---

##  常用命令

### 快速测试 (5分钟)

```bash
python scripts/train.py --config config_training.yaml --batch-size 2 --num-epochs 2
```

### 标准训练 (推荐)

```bash
python scripts/train.py --config config_training.yaml
```

### 自定义参数

```bash
python scripts/train.py \
    --config config_training.yaml \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --num-epochs 50
```

### 仅命令行参数

```bash
python scripts/train.py \
    --stage a \
    --batch-size 8 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --use-wandb
```

---

##  配置文件用法

### 方式 1: 直接使用配置文件

```bash
python scripts/train.py --config config_training.yaml
```

### 方式 2: 配置文件 + 命令行参数 (参数优先)

```bash
python scripts/train.py \
    --config config_training.yaml \
    --batch-size 16 \
    --learning-rate 5e-5
```

### 验证配置

```bash
python load_config.py --config config_training.yaml --validate
python load_config.py --config config_training.yaml --generate-command
```

---

##  数据概况

```
总样本: 3155
├─ Volcano:     1056 (33.5%)
├─ Earthquake:  1833 (58.1%)  ← 最多
├─ Wildfire:     142 (4.5%)
└─ Flood:        124 (3.9%)   ← 最少

不均衡系数: 14.76:1
→ 自动处理 (加权采样) 
```

---

##  文件位置

| 用途 | 路径 |
|------|------|
| 配置 | `config_training.yaml` |
| 训练 | `scripts/train.py` |
| 数据 | `data/` |
| 输出 | `outputs/checkpoints/` |
| 日志 | `outputs/logs/` |

---

## ⏱️ 预计时间

| 操作 | 时间 | GPU |
|------|------|-----|
| 验证 | 30秒 | 任何 |
| 数据加载 | 2分钟 | 任何 |
| Stage A (100 epochs) | 100分钟 | A100 |

---

##  常见问题

### Q: 如何使用配置文件?
```bash
python scripts/train.py --config config_training.yaml
```

### Q: 命令行参数如何覆盖配置?
```bash
python scripts/train.py --config config_training.yaml --batch-size 16
```

### Q: 如何检查配置是否正确?
```bash
python load_config.py --config config_training.yaml --show-config
```

### Q: 训练会保存什么?
```
outputs/
├── checkpoints/best_model.pt      (最优模型)
├── config_a.json                  (配置)
└── logs/                          (日志)
```

---

##  检查清单

启动前:
- [ ] 运行 `python quick_verify.py`
- [ ] 检查数据完整: `data/metadata.json` 存在
- [ ] CUDA 可用: `nvidia-smi`

启动后:
- [ ] 监控 W&B Dashboard
- [ ] 检查日志: `outputs/logs/`

---

##  下一步

-  详细架构说明: 查看 `ARCHITECTURE.md`
-  完整使用指南: 查看 `GUIDE.md`
-  配置参数详解: 查看 `config_training.yaml`


