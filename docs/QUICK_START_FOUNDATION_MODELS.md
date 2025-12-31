# Quick Start: Comparing Foundation Models

This is a quick reference guide for comparing different foundation models in CellViT++.

## TL;DR - Using UNI Instead of SAM-H

### Step 1: Get UNI Weights
```bash
# Download from HuggingFace
wget https://huggingface.co/mahmoodlab/UNI/resolve/main/pytorch_model.bin -O ./checkpoints/uni_model.bin
```

### Step 2: Create Config File

Copy an existing SAM config and modify:

```yaml
# config_uni.yaml
model:
  backbone: UNI  # Changed from SAM-H
  pretrained_encoder: ./checkpoints/uni_model.bin  # Changed path

training:
  batch_size: 8  # Adjust based on GPU memory
```

### Step 3: Train

```bash
python3 ./cellvit/train_cellvit.py --config ./configs/config_uni.yaml
```

### Step 4: Inference

```bash
python3 ./cellvit/detect_cells.py \
  --model ./logs/CellViT-UNI/checkpoints/model_best.pth \
  --outdir ./results/uni \
  process_wsi --wsi_path ./data/slide.svs
```

---

## Available Foundation Models

| Model | Backbone Parameter | Checkpoint Download | Memory (VRAM) |
|-------|-------------------|---------------------|---------------|
| **SAM-B** | `sam-b` | [Meta AI SAM](https://github.com/facebookresearch/segment-anything) | ~3GB |
| **SAM-L** | `sam-l` | [Meta AI SAM](https://github.com/facebookresearch/segment-anything) | ~5GB |
| **SAM-H** | `sam-h` | [Meta AI SAM](https://github.com/facebookresearch/segment-anything) | ~6GB |
| **UNI** | `uni` | [HuggingFace UNI](https://huggingface.co/mahmoodlab/UNI) | ~4GB |
| **Virchow** | `virchow` | [HuggingFace Virchow](https://huggingface.co/paige-ai/Virchow) | ~5GB |
| **Virchow2** | `virchow2` | [HuggingFace Virchow2](https://huggingface.co/paige-ai/Virchow2) | ~5GB |
| **ViT256** | `vit256` | [DINO ViT](https://github.com/facebookresearch/dino) | ~2GB |

---

## Model Comparison Script

Create `compare_foundations.sh`:

```bash
#!/bin/bash

# Array of models to compare
declare -A MODELS=(
  ["sam-h"]="./checkpoints/sam_vit_h.pth"
  ["uni"]="./checkpoints/uni_model.bin"
  ["virchow"]="./checkpoints/virchow_model.pth"
)

# Base config template
BASE_CONFIG="./configs/base_config.yaml"
DATASET_PATH="./data/pannuke"

for model_name in "${!MODELS[@]}"; do
  echo "========================================="
  echo "Training with $model_name"
  echo "========================================="
  
  # Create model-specific config
  cat > "./configs/train_${model_name}.yaml" << EOF
logging:
  project: CellViT-Foundation-Comparison
  log_comment: CellViT-${model_name}
  
model:
  backbone: ${model_name}
  pretrained_encoder: ${MODELS[$model_name]}

data:
  dataset_path: ${DATASET_PATH}
  # ... rest of your config
EOF

  # Train
  python3 ./cellvit/train_cellvit.py \
    --config "./configs/train_${model_name}.yaml"
done

echo "All models trained! Check ./logs/ for results"
```

---

## Config File Cheat Sheet

### Minimal Config for Foundation Model Swap

```yaml
model:
  backbone: UNI  # ← CHANGE THIS
  pretrained_encoder: /path/to/weights  # ← AND THIS

data:
  dataset_path: /path/to/dataset
  num_nuclei_classes: 6
  num_tissue_classes: 19

training:
  batch_size: 8  # ← ADJUST FOR YOUR GPU
  epochs: 130
  unfreeze_epoch: 25
  mixed_precision: true
```

### Full Config Template

See: `./docs/FOUNDATION_MODEL_GUIDE.md` for complete examples

---

## Understanding the Architecture

```
Input Image (1024x1024)
    ↓
┌───────────────────────┐
│  Foundation Model     │ ← Replace this part
│  (Frozen initially)   │
└───────────────────────┘
    ↓ Skip connections
┌───────────────────────┐
│  U-Net Decoder        │ ← Same for all models
└───────────────────────┘
    ↓
┌───────────────────────┐
│  Segmentation Heads   │ ← Same for all models
│  - Binary map         │
│  - HV map             │
│  - Type map           │
└───────────────────────┘
    ↓
Cell detections + classifications
```

---

## Common Issues & Solutions

### 1. CUDA Out of Memory
```yaml
training:
  batch_size: 4  # Reduce
  mixed_precision: true  # Enable
```

### 2. Checkpoint Not Loading
```python
# Check architecture in checkpoint
import torch
ckpt = torch.load("model.pth")
print(ckpt["arch"])  # Should be "CellViTUNI", "CellViTSAM", etc.
```

### 3. Wrong Input Size
Different models may have different expected input sizes. Check model file for `input_rescale_dict` or `img_size`.

---

## Performance Comparison Tips

### 1. Same Random Seed
```yaml
random_seed: 19  # Use same seed for all experiments
```

### 2. Same Hyperparameters
Keep everything the same except the backbone:
- Learning rate
- Batch size (if possible)
- Augmentations
- Loss weights

### 3. Track These Metrics
- Dice Score
- F1 Score
- AJI (Aggregated Jaccard Index)
- Inference time
- GPU memory usage

### 4. WandB Comparison
Use the same project and different groups:

```yaml
logging:
  project: Foundation-Model-Comparison
  group: ${model_name}  # Different group per model
```

Then compare in WandB dashboard.

---

## File Locations Reference

```
cellvit/
├── models/
│   ├── cell_segmentation/
│   │   ├── cellvit_sam.py      # SAM variants
│   │   ├── cellvit_uni.py      # UNI model
│   │   ├── cellvit_virchow.py  # Virchow model
│   │   └── backbones.py        # Encoder implementations
│   └── utils/
│       ├── sam_utils.py
│       ├── uni_utils.py
│       └── virchow_utils.py
├── training/
│   └── experiments/
│       └── experiment_cellvit_pannuke.py  # Training logic
└── inference/
    └── inference_disk.py       # Inference logic
```

---

## Next Steps

1. **Read the full guide**: `./docs/FOUNDATION_MODEL_GUIDE.md`
2. **Check example configs**: `./logs/PanNuke/CellViTHV/*/config.yaml`
3. **Download pretrained models**: From [Google Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79?usp=sharing)
4. **Try the example dataset**: `./test_database/training_database/Example-Detection`

---

## Questions?

- Check the full guide for detailed explanations
- Look at existing model implementations for reference
- See training configs in `./logs/` for working examples
