# CONDSAR NeDS - Quick Start Guide

**Get started with training in 5 minutes!**

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 12GB+ VRAM)
- 50GB+ free disk space

## Step 1: Install Dependencies

```bash
cd condsar

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install diffusers transformers accelerate
pip install pillow tifffile pyyaml
pip install wandb  # Optional, for experiment tracking
pip install tqdm
```

## Step 2: Verify Installation

```bash
python test_setup_neds.py
```

Expected output:
```
✅ All packages available
✅ CUDA available
✅ CONDSARNeDS model created
✅ Forward pass successful
🎉 All tests passed!
```

## Step 3: Prepare Your Data

### Option A: Use Existing Dataset

If you have data in `condsar/data/` with subdirectories `pre/`, `post/`, `mask/`:

```bash
python generate_metadata_neds.py --data_dir ./condsar/data
```

### Option B: Create Sample Data Structure

```bash
# Create directory structure
mkdir -p condsar/data/{pre,post,mask}

# Copy your images
cp /path/to/pre-disaster/images/* condsar/data/pre/
cp /path/to/post-disaster/sar/* condsar/data/post/
cp /path/to/masks/* condsar/data/mask/

# Generate metadata
python generate_metadata_neds.py --data_dir ./condsar/data
```

### Data Requirements

| Directory | Content | Format | Notes |
|-----------|---------|--------|-------|
| `pre/` | Pre-disaster RGB images | JPG/PNG/TIF | 512x512 recommended |
| `post/` | Post-disaster SAR images | TIF preferred | Same filenames as pre/ |
| `mask/` | Building/damage masks | TIF/PNG | Values: 0,1,2,3 |

**Important**: Files must have matching names:
- `pre/image001.jpg` ↔ `post/image001.tif` ↔ `mask/image001.tif`

## Step 4: Configure Training

Edit `config_neds.yaml` (or use defaults):

```yaml
data:
  source_dir: "./condsar/data"      # Your data path
  batch_size: 4                      # Reduce if OOM (try 2 or 1)

training:
  stage_a:
    num_epochs: 100                  # Number of training epochs
    learning_rate: 1.0e-4            # Learning rate

hardware:
  device: "cuda:0"                   # Use "cpu" if no GPU
```

## Step 5: Start Training!

### Basic Training

```bash
python train_neds.py --config config_neds.yaml --stage a
```

### Quick Test (5 epochs, no W&B)

```bash
python train_neds.py \
    --config config_neds.yaml \
    --stage a \
    --num_epochs 5 \
    --no_wandb
```

### Custom Settings

```bash
python train_neds.py \
    --config config_neds.yaml \
    --stage a \
    --batch_size 2 \
    --num_epochs 200 \
    --learning_rate 5e-5
```

## Step 6: Monitor Training

### Option A: Weights & Biases (Recommended)

1. Sign up at https://wandb.ai
2. Login: `wandb login`
3. Training dashboard will open automatically

### Option B: Local Logs

```bash
# View logs
tail -f logs/condsar_neds_*.log

# Or during training, watch the progress bar:
# Epoch 1/100: 100%|████████| 250/250 [02:30<00:00, loss=0.0234]
```

## Step 7: Check Results

### Checkpoints

Saved to `checkpoints/stage_a_neds/`:
```
checkpoint_epoch_10.pt
checkpoint_epoch_20.pt
...
final_model.pt
```

### Load Checkpoint for Inference

```python
import torch
from models.condsar_neds import CONDSARNeDS

model = CONDSARNeDS(...)
checkpoint = torch.load("checkpoints/stage_a_neds/final_model.pt")
model.load_state_dict(checkpoint['controlnet_state_dict'])
```

## Troubleshooting

### Issue: "metadata.json not found"

**Solution:**
```bash
python generate_metadata_neds.py --data_dir ./condsar/data
```

### Issue: "No matching files found"

**Solution:** Ensure files have matching names:
```bash
# Good ✅
pre/volcano_001.jpg
post/volcano_001.tif
mask/volcano_001.tif

# Bad ❌
pre/volcano_001.jpg
post/sar_001.tif      # Different name!
mask/mask_001.tif     # Different name!
```

### Issue: CUDA Out of Memory

**Solution 1:** Reduce batch size
```bash
python train_neds.py --config config_neds.yaml --stage a --batch_size 2
```

**Solution 2:** Use gradient accumulation
```yaml
# In config_neds.yaml
training:
  stage_a:
    batch_size: 2
    gradient_accumulation_steps: 2  # Effective batch = 4
```

**Solution 3:** Use CPU (very slow!)
```bash
python train_neds.py --config config_neds.yaml --stage a --device cpu
```

### Issue: Training is slow

**Solutions:**
1. Enable mixed precision (should be default)
   ```yaml
   training:
     stage_a:
       use_mixed_precision: true
   ```

2. Increase num_workers
   ```yaml
   data:
     num_workers: 8  # More workers = faster data loading
   ```

3. Use SSD for dataset storage (faster I/O)

### Issue: Cannot determine disaster type

**Solution:** Add disaster type to filename:
```bash
# Option 1: In filename
volcano_0.5_image001.jpg  # type=volcano, severity=0.5
earthquake_high_image002.jpg  # type=earthquake, severity=0.75

# Option 2: In directory structure
pre/volcano/image001.jpg
pre/earthquake/image002.jpg
```

## Expected Training Time

| GPU | Batch Size | Samples | Time per Epoch |
|-----|------------|---------|----------------|
| RTX 4090 | 8 | 1000 | ~5 min |
| RTX 3090 | 4 | 1000 | ~8 min |
| RTX 3080 | 2 | 1000 | ~15 min |
| V100 | 4 | 1000 | ~10 min |
| CPU | 1 | 1000 | ~2 hours ⚠️ |

100 epochs on RTX 3090 ≈ 13 hours

## Next Steps

Once training is complete:

1. **Evaluate model** (coming soon)
   ```bash
   python evaluate_neds.py --checkpoint checkpoints/stage_a_neds/final_model.pt
   ```

2. **Generate SAR images** (coming soon)
   ```bash
   python inference_neds.py \
       --checkpoint checkpoints/stage_a_neds/final_model.pt \
       --input pre_disaster.jpg \
       --mask building_mask.tif \
       --output generated_sar.tif
   ```

3. **Stage B: Generate synthetic data** (coming soon)

4. **Stage C: Mixed training** (coming soon)

## Tips for Best Results

1. **Data Quality**: Use high-quality, aligned image pairs
2. **Mask Accuracy**: Accurate masks → better results
3. **Disaster Type**: Include disaster type in filenames for better labeling
4. **Training Duration**: Train for at least 50-100 epochs
5. **Learning Rate**: Start with 1e-4, reduce if loss plateaus
6. **Batch Size**: Larger batch = more stable, but needs more memory
7. **Checkpoint Often**: Save every 10 epochs to avoid data loss

## Getting Help

1. Check logs: `tail -f logs/condsar_neds_*.log`
2. Review README_NEDS.md for detailed documentation
3. Open GitHub issue with error message + config
4. Check W&B dashboard for training curves

## Summary Checklist

- [ ] Install dependencies (`pip install ...`)
- [ ] Verify setup (`python test_setup_neds.py`)
- [ ] Prepare data (organize pre/post/mask)
- [ ] Generate metadata (`python generate_metadata_neds.py`)
- [ ] Configure training (`config_neds.yaml`)
- [ ] Start training (`python train_neds.py`)
- [ ] Monitor progress (W&B or logs)
- [ ] Check checkpoints after training

**Ready to train? Run:**
```bash
python train_neds.py --config config_neds.yaml --stage a
```

🚀 **Happy Training!**

