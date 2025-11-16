# ✅ 训练脚本调试完成总结

## 问题诊断和解决

### 问题 1: metadata.json 路径不匹配
**症状**: `FileNotFoundError: metadata.json not found in data\source`

**根本原因**:
- 配置文件和代码中默认路径设置为 `./data/source` 和 `./data/target`
- 实际数据位于 `./data` 目录

**解决方案**:
修改了以下文件的默认路径：
- `config_training.yaml`: 将 `source_dir` 和 `target_dir` 改为 `./data`
- `scripts/train.py`: TrainingConfig 中的默认值改为 `./data`
- `load_config.py`: 默认值改为 `./data`
- `models/train_pipeline.py`: 所有 Stage 的数据路径改为 `./data`
- `generate_metadata.py`: 路径改为使用正斜杠 `/`

### 问题 2: ControlNetModel 导入错误
**症状**: `TypeError: ControlNetModel.__init__() got an unexpected keyword argument 'embedding_dim'`

**根本原因**:
- `EnhancedDisasterControlNet` 继承自 `ControlNetModel`
- 在初始化时传入了 `ControlNetModel` 不支持的参数

**解决方案**:
- 修改 `models/enhanced_condsar.py` 改为不直接继承 `ControlNetModel`，而是使用组合模式
- 创建自定义的条件编码器和特征融合层
- 简化模型架构以避免依赖 diffusers 的内部实现

### 问题 3: 模块导入问题
**症状**: 
- `ModuleNotFoundError: No module named 'diffusers.models'`
- `ImportError: cannot import name 'StableDiffusionPipeline' from 'diffusers'`

**根本原因**:
- diffusers 库的导入路径不同版本差异大
- 相对导入问题

**解决方案**:
- `models/enhanced_condsar.py`: 改为 `from diffusers import ControlNetModel`
- `models/training_stage_a.py`: 添加了 fallback 导入逻辑和相对导入支持
- 移除未使用的导入（如 `StageATrainer`）

### 问题 4: 配置参数为 None
**症状**: `TypeError: device() received an invalid combination of arguments - got (NoneType)`

**根本原因**:
- 命令行参数的默认值是 `None`，覆盖了配置文件中的值
- `TrainingConfig` 使用 `kwargs.get()` 但没有处理 `None` 值

**解决方案**:
- `scripts/train.py` 中 `merge_config_with_args()` 函数添加了默认值保证
- `TrainingConfig.__init__()` 中所有参数改为使用 `or` 操作符确保不会是 `None`

## ✅ 当前状态

训练脚本现在能够正常启动！输出显示：

```
2025-11-17 07:58:56 - condsar_trainer - INFO - Loading source dataset from ./data
2025-11-17 07:58:56 - condsar_trainer - INFO - Loaded 3155 images from metadata.json
2025-11-17 07:58:56 - condsar_trainer - INFO - Disaster types: ['Volcano', 'Earthquake', 'Wildfire', 'Flood']
2025-11-17 07:58:56 - condsar_trainer - INFO - Loaded 3155 training samples
2025-11-17 07:58:56 - condsar_trainer - INFO - Batch size: 1
2025-11-17 07:58:56 - condsar_trainer - INFO - Total batches: 3155
2025-11-17 07:58:56 - condsar_trainer - INFO - Creating EnhancedDisasterControlNet...
2025-11-17 07:58:58 - condsar_trainer - INFO - Model created with 379,647,296 parameters
2025-11-17 07:58:58 - condsar_trainer - INFO - Creating SAR VAE Decoder...
2025-11-17 07:58:58 - condsar_trainer - INFO - SAR VAE Decoder created with 2,656,257 parameters
```

## 文件修改清单

| 文件 | 修改内容 |
|------|--------|
| `config_training.yaml` | source_dir/target_dir 改为 ./data |
| `scripts/train.py` | device/output_dir 等参数 None 检查，默认值设置 |
| `generate_metadata.py` | 路径改为正斜杠 |
| `load_config.py` | source_dir 默认值改为 ./data |
| `models/train_pipeline.py` | 所有 Stage 的数据路径改为 ./data |
| `models/enhanced_condsar.py` | 改进导入、简化模型初始化 |
| `models/training_stage_a.py` | 修复导入、添加 fallback 逻辑 |

## 下一步建议

1. **日志编码问题**: 如果需要在 Windows 上更好地显示 Unicode 字符，可以在 logger 配置中指定 UTF-8 编码
2. **训练参数**: 调整 `--batch-size` 和 `--num-epochs` 来适应你的硬件
3. **模型权重**: 确保有足够的显存（当前 379M 参数的 ControlNet + 2.6M VAE Decoder）

## 使用方法

```bash
cd D:\condsar

# 方式1: 直接使用命令行参数
python scripts/train.py --stage a --num-epochs 100 --batch-size 4

# 方式2: 使用配置文件
python scripts/train.py --config config_training.yaml --stage a
```


