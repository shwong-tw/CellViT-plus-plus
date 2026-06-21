# STHELAR Dataset Integration for CellViT++

This directory contains documentation, processing scripts, and training configurations
for using the [STHELAR dataset](https://huggingface.co/datasets/FelicieGS/STHELAR_40x)
with CellViT++.

## Quick Start

### 1. Install Additional Dependencies

```bash
pip install datasets scipy huggingface_hub pyyaml
```

### 2. Download and Preprocess the Dataset

```bash
# 5-class setup (recommended for initial experiments)
python -m cellvit.training.datasets.sthelar.prepare_sthelar \
    --output_dir /data/STHELAR_5class \
    --magnification 40x \
    --num_classes 5 \
    --split_strategy spatial \
    --max_patches_per_slide 5000

# 9-class fine-grained setup
python -m cellvit.training.datasets.sthelar.prepare_sthelar \
    --output_dir /data/STHELAR_9class \
    --magnification 40x \
    --num_classes 9 \
    --split_strategy slide

# Cancer vs Normal (2-class) setup
python -m cellvit.training.datasets.sthelar.prepare_sthelar \
    --output_dir /data/STHELAR_cancer \
    --magnification 40x \
    --num_classes 2 \
    --split_strategy slide \
    --tissues breast lung pancreatic
```

### 3. Train a Cell Classifier

```bash
# Edit the config to set your dataset and checkpoint paths
# Then run:
python cellvit/train_cell_classifier_head.py \
    --config docs/sthelar/configs/sthelar_classifier_5class.yaml
```

## Dataset Overview

**STHELAR** (Spatial Transcriptomics and H&E Linked Annotations Resource) provides:
- ~11M+ annotated cells across 31 slides and 16 tissue types
- Cell type annotations derived from Xenium spatial transcriptomics
- Co-registered H&E patches at 20x and 40x magnification
- 256×256 pixel patches with per-cell instance segmentation and type labels

### Label Spaces

| Setup | Classes | Use Case |
|-------|---------|----------|
| 5-class | Immune, Stromal, Epithelial, Other (+Background) | General cell classification |
| 9-class | Epithelial, Blood_vessel, Fibroblast, Myeloid, B_Plasma, T_NK, Melanocyte, Specialized, Other | Fine-grained typing |
| 2-class | Cancer, Normal (+Background) | Tumor detection |

### Split Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `spatial` | Coordinate-based with boundary margin | Single-slide experiments |
| `slide` | Whole-slide assignment | Multi-slide experiments (recommended) |
| `random` | Random shuffle | Baseline comparison only |
| `auto` | Slide if ≥2 slides, spatial otherwise | Default |

## File Structure

```
docs/sthelar/
├── README.md                          # This file
├── tutorial_sthelar_repos.md          # Detailed explanation of STHELAR repos
└── configs/
    ├── sthelar_classifier_5class.yaml # 5-class training config
    └── sthelar_classifier_9class.yaml # 9-class training config

cellvit/training/datasets/sthelar/
├── __init__.py                        # STHELARDataset class
└── prepare_sthelar.py                 # Download + preprocessing script
```

## Processing Pipeline

```
HuggingFace (FelicieGS/STHELAR_40x)
        │
        ▼
prepare_sthelar.py
  ├── Download parquet shards
  ├── Decode images (PNG → RGB numpy)
  ├── Decode cell_id_map (sparse CSR → dense int32)
  ├── Load cell_metadata per slide
  ├── Build type_map (cell_id → label → class int)
  ├── Extract centroids + types
  ├── Assign splits (spatial/slide/random)
  └── Save in CellViT++ format
        │
        ▼
Processed Dataset (on disk)
  ├── train/images/*.png
  ├── train/detections/*.json
  ├── val/images/*.png
  ├── val/detections/*.json
  └── dataset_config.yaml
        │
        ▼
train_cell_classifier_head.py
  ├── Load STHELARDataset
  ├── Load pretrained CellViT model
  ├── Create LinearClassifier head
  ├── Train with CellViTHeadTrainer
  └── Save best model checkpoint
```

## Advanced Usage

### Filtering by Tissue Type

```bash
# Only breast and lung tissues
python -m cellvit.training.datasets.sthelar.prepare_sthelar \
    --output_dir /data/STHELAR_breast_lung \
    --tissues breast lung \
    --split_strategy slide
```

### Using Different Backbones

Update the config YAML:
```yaml
# For UNI backbone (1024-dim embeddings)
cellvit_path: ./checkpoints/CellViT-UNI-x40-AMP.pth

# For SAM-H backbone (1280-dim embeddings)
cellvit_path: ./checkpoints/CellViT-SAM-H-x40-AMP.pth
```

### Resuming Training

```bash
python cellvit/train_cell_classifier_head.py \
    --config docs/sthelar/configs/sthelar_classifier_5class.yaml \
    --checkpoint /path/to/checkpoint.pth
```

## References

- **STHELAR Paper**: Scientific Data, 2026
- **Dataset (40x)**: https://huggingface.co/datasets/FelicieGS/STHELAR_40x
- **Dataset (20x)**: https://huggingface.co/datasets/FelicieGS/STHELAR_20x
- **STHELAR Pipeline**: https://github.com/MICS-Lab/STHELAR
- **CellViT for STHELAR**: https://github.com/MICS-Lab/CellViT_for_STHELAR
- **BioImage Archive**: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2146
