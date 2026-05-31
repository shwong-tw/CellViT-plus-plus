# Foundation Model Replacement Guide

This guide explains how to replace or add different foundation models to CellViT++ for comparing performance across different vision transformer backbones.

## Table of Contents
1. [Understanding the Model Architecture](#understanding-the-model-architecture)
2. [How Foundation Models are Integrated](#how-foundation-models-are-integrated)
3. [Step-by-Step Example: Using UNI](#step-by-step-example-using-uni)
4. [Adding a New Foundation Model](#adding-a-new-foundation-model)
5. [Configuration Parameters](#configuration-parameters)
6. [Inference with Different Foundation Models](#inference-with-different-foundation-models)

---

## Understanding the Model Architecture

CellViT++ uses a **U-Net-like architecture** with a **Vision Transformer (ViT) backbone** as the encoder. The framework is designed to be modular, allowing you to swap different foundation models as the encoder while keeping the decoder and segmentation heads consistent.

### Key Components

```
┌─────────────────────────────────────────────────┐
│           CellViT++ Architecture                │
├─────────────────────────────────────────────────┤
│                                                 │
│  Input Image (256×256 or 1024×1024)            │
│          ↓                                      │
│  ┌──────────────────────────┐                  │
│  │  Foundation Model        │                  │
│  │  (Encoder/Backbone)      │  ← REPLACEABLE   │
│  │  - SAM-H, SAM-L, SAM-B   │                  │
│  │  - UNI                   │                  │
│  │  - Virchow, Virchow2     │                  │
│  │  - ViT256                │                  │
│  │  - HIPT                  │                  │
│  └──────────────────────────┘                  │
│          ↓                                      │
│  Skip Connections (Multi-scale features)        │
│          ↓                                      │
│  ┌──────────────────────────┐                  │
│  │  Decoder (U-Net style)   │  ← FIXED         │
│  └──────────────────────────┘                  │
│          ↓                                      │
│  ┌──────────────────────────┐                  │
│  │  Segmentation Heads      │  ← FIXED         │
│  │  - Binary Map            │                  │
│  │  - HV Map                │                  │
│  │  - Nuclei Type Map       │                  │
│  └──────────────────────────┘                  │
│          ↓                                      │
│  Cell Detection & Classification                │
└─────────────────────────────────────────────────┘
```

### Foundation Models Location

Foundation model implementations are in:
- **Models**: `/cellvit/models/cell_segmentation/`
  - `cellvit_sam.py` - SAM variants (B, L, H)
  - `cellvit_uni.py` - UNI foundation model
  - `cellvit_virchow.py` - Virchow foundation model
  - `cellvit_virchow2.py` - Virchow2 foundation model
  - `cellvit_256.py` - ViT256 foundation model

- **Backbones**: `/cellvit/models/cell_segmentation/backbones.py`
  - Contains the actual encoder implementations

- **Utilities**: `/cellvit/models/utils/`
  - `sam_utils.py` - SAM-specific utilities
  - `uni_utils.py` - UNI-specific utilities
  - `virchow_utils.py` - Virchow-specific utilities

---

## How Foundation Models are Integrated

### 1. Model Class Structure

Each foundation model has its own wrapper class that inherits from the base `CellViT` class:

```python
class CellViTUNI(CellViT):
    """CellViT with UNI backbone settings"""
    
    def __init__(self, model_uni_path, num_nuclei_classes, num_tissue_classes, ...):
        # Set foundation model specific parameters
        self.embed_dim = 1024      # UNI embedding dimension
        self.depth = 24            # Number of transformer blocks
        self.num_heads = 12        # Attention heads
        self.extract_layers = [6, 12, 18, 24]  # Skip connection layers
        
        # Initialize base CellViT with these parameters
        super().__init__(...)
        
        # Load the foundation model encoder
        self.encoder = ViTCellViTUNI(...)
        self.load_pretrained_encoder(model_uni_path)
```

### 2. Training Integration

During training, the foundation model is specified via the `backbone` parameter in the configuration file:

```yaml
model:
  backbone: UNI  # Options: default, vit256, sam-b, sam-l, sam-h, uni, virchow, virchow2
  pretrained_encoder: /path/to/uni_checkpoint.pth
```

The training script (`experiment_cellvit_pannuke.py`) handles model instantiation:

```python
def get_train_model(self, backbone_type: str, pretrained_encoder: str, ...):
    if backbone_type.lower() == "uni":
        model = CellViTUNI(
            model_uni_path=pretrained_encoder,
            num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
        )
        model.freeze_encoder()  # Freeze foundation model weights
```

### 3. Inference Integration

During inference, the model architecture is automatically detected from the checkpoint:

```python
# In inference_disk.py
model_checkpoint = torch.load(model_path)
model = self._get_model(model_type=model_checkpoint["arch"])
# "arch" will be "CellViTUNI", "CellViTSAM", etc.
```

---

## Step-by-Step Example: Using UNI

### Prerequisites

1. **Download UNI checkpoint**:
   - Visit HuggingFace: https://huggingface.co/mahmoodlab/UNI
   - Download the model weights (e.g., `pytorch_model.bin`)
   - Place in `./checkpoints/` directory

2. **System Requirements**:
   - GPU with at least 24GB VRAM (UNI is a large model)
   - PyTorch 2.2+ with CUDA support

### Training with UNI

#### Step 1: Create Configuration File

Create a YAML configuration file (e.g., `train_cellvit_uni.yaml`):

```yaml
logging:
  mode: online  # or 'offline' for local-only
  project: CellViT-UNI-Comparison
  notes: Training CellViT with UNI backbone
  log_comment: CellViT-UNI-Fold-1
  wandb_dir: ./logs
  level: Debug

random_seed: 19
gpu: 0

data:
  dataset: PanNuke  # or your custom dataset
  dataset_path: /path/to/pannuke/dataset
  train_folds: [0]
  val_folds: [1]
  test_folds: [2]
  num_nuclei_classes: 6
  num_tissue_classes: 19

model:
  backbone: UNI  # ← THIS IS THE KEY PARAMETER
  pretrained_encoder: ./checkpoints/uni_pytorch_model.bin  # ← PATH TO UNI WEIGHTS
  shared_skip_connections: true

loss:
  nuclei_binary_map:
    focaltverskyloss:
      loss_fn: FocalTverskyLoss
      weight: 1
    dice:
      loss_fn: dice_loss
      weight: 1
  hv_map:
    mse:
      loss_fn: mse_loss_maps
      weight: 2.5
    msge:
      loss_fn: msge_loss_maps
      weight: 8
  nuclei_type_map:
    bce:
      loss_fn: xentropy_loss
      weight: 0.5
    dice:
      loss_fn: dice_loss
      weight: 0.2
  tissue_types:
    ce:
      loss_fn: CrossEntropyLoss
      weight: 0.1

training:
  drop_rate: 0
  attn_drop_rate: 0.1
  drop_path_rate: 0.1
  batch_size: 8  # Reduce if memory issues occur
  epochs: 130
  optimizer: AdamW
  early_stopping_patience: 130
  scheduler:
    scheduler_type: exponential
    hyperparameters:
      gamma: 0.85
  optimizer_hyperparameter:
    betas: [0.85, 0.95]
    lr: 0.0003
    weight_decay: 0.0001
  unfreeze_epoch: 25  # Encoder stays frozen until this epoch
  sampling_gamma: 0.85
  sampling_strategy: cell+tissue
  mixed_precision: true  # Recommended for memory efficiency
```

#### Step 2: Run Training

```bash
python3 ./cellvit/train_cellvit.py \
  --config ./configs/train_cellvit_uni.yaml
```

#### Step 3: Monitor Training

- Check WandB dashboard for metrics
- Training logs will be in `./logs/`
- Checkpoints saved in `./logs/<run_name>/checkpoints/`

### Inference with UNI Model

Once trained, use the checkpoint for inference:

```bash
python3 ./cellvit/detect_cells.py \
  --model ./logs/CellViT-UNI-Fold-1/checkpoints/model_best.pth \
  --outdir ./results/uni_inference \
  --geojson \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
```

---

## Adding a New Foundation Model

To add a completely new foundation model (e.g., a new ViT variant), follow these steps:

### Step 1: Create Model Utilities (if needed)

Create a utility file in `/cellvit/models/utils/` for model-specific code:

```python
# /cellvit/models/utils/new_foundation_utils.py

import torch
import torch.nn as nn

class NewFoundationEncoder(nn.Module):
    """
    Wrapper for your foundation model encoder
    """
    def __init__(self, ...):
        super().__init__()
        # Load your foundation model architecture
        
    def forward(self, x):
        # Forward pass
        return features
```

### Step 2: Create Backbone Wrapper

Add a backbone class in `/cellvit/models/cell_segmentation/backbones.py`:

```python
class ViTCellViTNewFoundation(nn.Module):
    """Backbone wrapper for new foundation model"""
    
    def __init__(self, extract_layers, num_classes=0):
        super().__init__()
        self.extract_layers = extract_layers
        # Initialize your foundation model
        from cellvit.models.utils.new_foundation_utils import NewFoundationEncoder
        self.model = NewFoundationEncoder(...)
        
    def forward(self, x):
        """Extract features at specified layers"""
        extracted_layers = []
        # Your forward pass logic
        return x, None, extracted_layers
```

### Step 3: Create CellViT Variant

Create a new file `/cellvit/models/cell_segmentation/cellvit_newfoundation.py`:

```python
from pathlib import Path
from typing import Union
import torch
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.backbones import ViTCellViTNewFoundation

class CellViTNewFoundation(CellViT):
    """CellViT with NewFoundation backbone"""
    
    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        # Set architecture-specific parameters
        self.img_size = 224
        self.patch_size = 16
        self.embed_dim = 768  # Adjust for your model
        self.depth = 12       # Adjust for your model
        self.num_heads = 12   # Adjust for your model
        self.extract_layers = [3, 6, 9, 12]  # Which layers to extract
        self.input_channels = 3
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.model_path = model_path
        
        # Initialize parent
        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            regression_loss=False,
        )
        
        # Create encoder
        self.encoder = ViTCellViTNewFoundation(
            extract_layers=self.extract_layers,
            num_classes=num_tissue_classes
        )
        
        # Load pretrained weights
        self.load_pretrained_encoder(self.model_path)
    
    def load_pretrained_encoder(self, model_path: Union[Path, str]):
        """Load pretrained weights"""
        if model_path is None:
            print("No checkpoint provided!")
        else:
            state_dict = torch.load(str(model_path), map_location="cpu")
            # You may need to adapt keys or use strict=False
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loading checkpoint: {msg}")
```

### Step 4: Register in Training Script

Update `/cellvit/training/experiments/experiment_cellvit_pannuke.py`:

```python
# Add to imports
from cellvit.models.cell_segmentation.cellvit_newfoundation import CellViTNewFoundation

# In get_train_model method, add to implemented_backbones list:
implemented_backbones = [
    "default",
    "vit256",
    "sam-b", "sam-l", "sam-h",
    "uni",
    "virchow", "virchow2",
    "newfoundation",  # ← Add your model
]

# Add model instantiation:
if backbone_type.lower() == "newfoundation":
    model = CellViTNewFoundation(
        model_path=pretrained_encoder,
        num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
        num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
    )
    if pretrained_model is not None:
        cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
        self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
    model.freeze_encoder()
    self.logger.info(f"Loaded CellViTNewFoundation model")
```

### Step 5: Register in Inference Script

Update `/cellvit/inference/inference_disk.py`:

```python
# Add to imports
from cellvit.models.cell_segmentation.cellvit_newfoundation import CellViTNewFoundation

# In _get_model method:
implemented_models = ["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI", "CellViTNewFoundation"]

# Add model case:
elif model_type == "CellViTNewFoundation":
    model = CellViTNewFoundation(
        model_path=None,
        num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
        num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
    )
```

### Step 6: Test Your Model

Create a test configuration and verify:

```bash
# Test training
python3 ./cellvit/train_cellvit.py --config ./configs/test_newfoundation.yaml

# Test inference
python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/cellvit_newfoundation.pth \
  --outdir ./results \
  process_wsi --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
```

---

## Configuration Parameters

### Key Parameters to Modify for Different Foundation Models

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `model.backbone` | Foundation model type | `"SAM-H"`, `"UNI"`, `"Virchow"`, `"ViT256"` |
| `model.pretrained_encoder` | Path to pretrained weights | `"./checkpoints/uni_model.bin"` |
| `model.embed_dim` | Encoder embedding dimension | `768`, `1024`, `1280` |
| `model.depth` | Number of transformer layers | `12`, `24`, `32` |
| `model.num_heads` | Number of attention heads | `12`, `16` |
| `model.extract_layers` | Layers for skip connections | `[3, 6, 9, 12]`, `[8, 16, 24, 32]` |
| `training.batch_size` | Batch size (adjust for GPU) | `8`, `16`, `32` |
| `training.unfreeze_epoch` | When to unfreeze encoder | `25`, `50`, `never` |
| `training.mixed_precision` | Use AMP for efficiency | `true`, `false` |

### Foundation Model Specifications

| Model | embed_dim | depth | num_heads | extract_layers | Notes |
|-------|-----------|-------|-----------|----------------|-------|
| **SAM-B** | 768 | 12 | 12 | [3, 6, 9, 12] | Base SAM |
| **SAM-L** | 1024 | 24 | 16 | [6, 12, 18, 24] | Large SAM |
| **SAM-H** | 1280 | 32 | 16 | [8, 16, 24, 32] | Huge SAM |
| **UNI** | 1024 | 24 | 12 | [6, 12, 18, 24] | PathologyFoundation |
| **Virchow** | 1280 | 32 | 16 | [8, 16, 24, 32] | Paige.AI model |
| **Virchow2** | 1280 | 32 | 16 | [8, 16, 24, 32] | Paige.AI v2 |
| **ViT256** | 384 | 12 | 6 | [3, 6, 9, 12] | DINO pretrained |

---

## Inference with Different Foundation Models

### Using Pre-trained Checkpoints

The authors provide checkpoints for different foundation models. Download from [Google Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79?usp=sharing).

```bash
# SAM-H model
python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --outdir ./results/sam_h \
  process_wsi --wsi_path ./data/slide.svs

# UNI model (if you trained it)
python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-UNI-x40.pth \
  --outdir ./results/uni \
  process_wsi --wsi_path ./data/slide.svs

# Virchow model
python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-Virchow-x40.pth \
  --outdir ./results/virchow \
  process_wsi --wsi_path ./data/slide.svs
```

### Comparing Multiple Models

Create a script to run inference with multiple models:

```bash
#!/bin/bash
# compare_models.sh

MODELS=(
  "./checkpoints/CellViT-SAM-H-x40-AMP.pth"
  "./checkpoints/CellViT-UNI-x40.pth"
  "./checkpoints/CellViT-Virchow-x40.pth"
)

NAMES=("sam_h" "uni" "virchow")
WSI_PATH="./data/test_slide.svs"

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  NAME="${NAMES[$i]}"
  
  echo "Running inference with $NAME..."
  python3 ./cellvit/detect_cells.py \
    --model "$MODEL" \
    --outdir "./results/comparison/$NAME" \
    --geojson \
    --graph \
    process_wsi --wsi_path "$WSI_PATH"
done

echo "Comparison complete! Check ./results/comparison/"
```

---

## Tips and Best Practices

### 1. Memory Management

Different foundation models have different memory requirements:

- **SAM-H**: Largest, ~6GB VRAM per image
- **Virchow**: Large, ~5GB VRAM per image
- **UNI**: Medium, ~4GB VRAM per image
- **ViT256**: Smallest, ~2GB VRAM per image

**Adjust batch size accordingly**:
```yaml
training:
  batch_size: 4   # For SAM-H, Virchow on 24GB GPU
  batch_size: 8   # For UNI on 24GB GPU
  batch_size: 16  # For ViT256 on 24GB GPU
```

### 2. Transfer Learning Strategy

Foundation models are typically kept frozen during initial training:

```yaml
training:
  unfreeze_epoch: 25  # Keep encoder frozen for 25 epochs
```

This prevents catastrophic forgetting of pretrained features.

### 3. Checkpoint Organization

Organize checkpoints by foundation model:

```
checkpoints/
├── foundation_models/
│   ├── sam_vit_h.pth
│   ├── uni_pytorch_model.bin
│   └── virchow_model.pth
├── cellvit_trained/
│   ├── sam_h/
│   │   └── model_best.pth
│   ├── uni/
│   │   └── model_best.pth
│   └── virchow/
│       └── model_best.pth
└── classifiers/
    ├── sam-h/
    ├── uni/
    └── virchow/
```

### 4. Reproducibility

Always set random seeds for fair comparison:

```yaml
random_seed: 19  # Same seed across all experiments
```

### 5. Evaluation Metrics

Track these metrics for comparison:
- **Dice Score**: Segmentation quality
- **F1 Score**: Detection accuracy
- **AJI (Aggregated Jaccard Index)**: Instance segmentation
- **Inference Time**: Speed comparison
- **Memory Usage**: Resource efficiency

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use mixed precision:
```yaml
training:
  batch_size: 4
  mixed_precision: true
```

### Issue: Checkpoint Loading Fails

**Solution**: Check architecture name and ensure compatibility:
```python
checkpoint = torch.load("model.pth")
print(checkpoint["arch"])  # Should match your model class
```

### Issue: Different Input Sizes

**Solution**: Some models (like Virchow) require specific input sizes. Check model documentation:
```python
# In cellvit_virchow.py
self.input_rescale_dict = {256: 252, 1024: 1022}
```

---

## Summary

To replace the foundation model in CellViT++:

1. **For existing models** (SAM, UNI, Virchow): Just change the `backbone` parameter in config
2. **For new models**: Follow the 6-step process to integrate the architecture
3. **For comparison**: Train multiple models with same config but different backbones
4. **For inference**: Use the trained checkpoints with `detect_cells.py`

The modular design makes it straightforward to experiment with different foundation models while maintaining consistent decoder and task heads for fair comparison.

---

## References

- **SAM**: [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- **UNI**: [A General-Purpose Self-Supervised Model for Computational Pathology](https://github.com/mahmoodlab/UNI)
- **Virchow**: [Paige.AI Foundation Model](https://huggingface.co/paige-ai/Virchow)
- **CellViT++**: [Paper on arXiv](https://arxiv.org/abs/2501.05269)
