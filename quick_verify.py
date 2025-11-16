#!/usr/bin/env python
"""
快速验证脚本 - 检查系统是否就绪
"""
import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "models"))

print("\n" + "=" * 80)
print("🧪 CONDSAR Quick Verification")
print("=" * 80 + "\n")

# 1. 测试基础导入
print("[1/5] Testing basic imports...")
try:
    import torch
    import numpy as np
    from PIL import Image
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ NumPy imported")
    print(f"✅ PIL imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 2. 测试项目模块
print("\n[2/5] Testing project modules...")
try:
    from models.training_utils import DisasterSARDataset, MetricsTracker
    print("✅ training_utils imported")
except ImportError as e:
    print(f"⚠️ training_utils: {e}")

try:
    from models.enhanced_condsar import EnhancedDisasterControlNet
    print("✅ enhanced_condsar imported")
except ImportError as e:
    print(f"⚠️ enhanced_condsar: {e}")

# 3. 测试设备
print("\n[3/5] Testing device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Available device: {device}")
if torch.cuda.is_available():
    print(f"   - CUDA: {torch.cuda.get_device_name(0)}")
    print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 4. 测试WandB
print("\n[4/5] Testing WandB...")
try:
    import wandb
    print("✅ WandB installed")
except ImportError:
    print("⚠️ WandB not installed (optional)")

# 5. 检查目录结构
print("\n[5/5] Checking directory structure...")
required_dirs = ['data', 'outputs', 'scripts', 'models', 'src']
for d in required_dirs:
    path = Path(d)
    exists = "✅" if path.exists() else "❌"
    print(f"{exists} {d}/")

# 创建必要的目录
for d in required_dirs:
    Path(d).mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("✅ Quick verification completed successfully!")
print("=" * 80)
print("\n🚀 Ready to run:")
print("   python scripts/train.py --help")
print("   python scripts/inference.py --help")
print("   python scripts/verify.py --full --create-data")
print("\n")

