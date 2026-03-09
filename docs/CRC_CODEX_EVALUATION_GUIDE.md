# CRC CODEX Dataset Evaluation Guide

## Overview

This guide explains how to evaluate your CRC CODEX classifier using the custom evaluation script and dataset class created specifically for your dataset format.

## Dataset Format

Your CRC CODEX dataset has a specific structure with **256x256 pixel patches**:

### Directory Structure
```
dataset_path/
├── {split}/              # e.g., test, val, train
│   ├── images/
│   │   └── *.npy        # Image files as numpy arrays
│   └── labels/
│       └── *.npy        # Label files as dictionaries
```

### Image Format

Images are stored as `.npy` files:
- **Size**: 256×256 pixels
- **Format**: NumPy arrays (can be uint8 or normalized float)
- **Channels**: RGB (3 channels) or grayscale (auto-converted to RGB)

### Label Format

Labels are stored as `.npy` files containing dictionaries:

```python
label = np.load(label_path, allow_pickle=True)
label_dict = label.item()  # Get dictionary from numpy array

# Dictionary contains:
inst_map = label_dict.get("inst_map")  # Instance segmentation map (256, 256)
type_map = label_dict.get("type_map")  # Nuclei type map (256, 256)
```

### Nuclei Types

Your dataset has **3 nuclei classes**:
- **0**: Background (implicit)
- **1**: Connective
- **2**: Inflammatory
- **3**: Neoplastic

---

## Quick Start

### 1. Verify Your Dataset Structure

```bash
# Your dataset should look like this:
ls ./training_data/crc_codex/256x256_160x160/test/images/*.npy
ls ./training_data/crc_codex/256x256_160x160/test/labels/*.npy
```

### 2. Run Evaluation

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir ./log_local/2026-02-06T205610_cellvit++ \
  --cellvit_path ./checkpoints/HIPT-25/CellViT-256-x40-AMP.pth \
  --dataset_path ./training_data/crc_codex/256x256_160x160 \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

### 3. Check Results

Results will be saved to:
```
./log_local/2026-02-06T205610_cellvit++/inference_results.json
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--logdir` | ✅ Yes | - | Path to training log directory with classifier |
| `--cellvit_path` | ✅ Yes | - | Path to pretrained CellViT model (.pth) |
| `--dataset_path` | ✅ Yes | - | Path to CRC CODEX dataset root |
| `--split` | No | `test` | Dataset split name (test/val/train) |
| `--checkpoint_name` | No | `model_best.pth` | Classifier checkpoint filename |
| `--gpu` | No | `0` | GPU ID to use |
| `--magnification` | No | `40` | Magnification level (20 or 40) |

---

## Output Metrics

The script calculates comprehensive metrics:

### Overall Binary Metrics
- **Dice**: Dice coefficient for binary segmentation
- **AJI**: Aggregated Jaccard Index
- **AJI+**: Strict version of AJI
- **PQ**: Panoptic Quality
- **DQ**: Detection Quality
- **SQ**: Segmentation Quality

### Per-Nuclei-Type Metrics

For each of the 3 nuclei types (Connective, Inflammatory, Neoplastic):
- **Dice**: Type-specific Dice coefficient
- **AJI**: Type-specific Aggregated Jaccard Index
- **PQ, DQ, SQ**: Type-specific panoptic quality metrics
- **F1**: Cell-level F1 score
- **Precision**: Cell-level precision
- **Recall**: Cell-level recall

---

## Console Output Example

```
================================================================================
Overall Metrics:
================================================================================
DICE: 0.8523 ± 0.0234
AJI: 0.7891 ± 0.0312
AJI_PLUS: 0.7456 ± 0.0298
PQ: 0.7234 ± 0.0276
DQ: 0.8012 ± 0.0245
SQ: 0.9023 ± 0.0156

================================================================================
Per-Nuclei-Type Metrics:
================================================================================

Connective (Type 1):
----------------------------------------------------------------------
  dice: 0.8234 ± 0.0312
  aji: 0.7456 ± 0.0387
  pq: 0.6923 ± 0.0345
  dq: 0.7712 ± 0.0298
  sq: 0.8956 ± 0.0187
  f1_cell: 0.8123 ± 0.0267
  prec_cell: 0.8456 ± 0.0245
  rec_cell: 0.7812 ± 0.0289

Inflammatory (Type 2):
----------------------------------------------------------------------
  dice: 0.8645 ± 0.0245
  ...

Neoplastic (Type 3):
----------------------------------------------------------------------
  dice: 0.8701 ± 0.0198
  ...
```

---

## JSON Output Structure

```json
{
  "overall": {
    "dice": 0.8523,
    "aji": 0.7891,
    "aji_plus": 0.7456,
    "pq": 0.7234,
    "dq": 0.8012,
    "sq": 0.9023
  },
  "per_type": {
    "Connective": {
      "dice": 0.8234,
      "aji": 0.7456,
      "pq": 0.6923,
      "dq": 0.7712,
      "sq": 0.8956,
      "f1_cell": 0.8123,
      "prec_cell": 0.8456,
      "rec_cell": 0.7812
    },
    "Inflammatory": { ... },
    "Neoplastic": { ... }
  }
}
```

---

## Understanding the Evaluation Process

### Step 1: Data Loading
- Loads images and labels from `.npy` files
- Extracts `inst_map` and `type_map` from dictionary
- Caches dataset in memory for faster processing

### Step 2: CellViT Inference
- Runs CellViT model to detect and segment nuclei
- Extracts cell embeddings for each detected nucleus

### Step 3: Classification
- Runs your trained classifier on each cell embedding
- Predicts nuclei type (Connective, Inflammatory, Neoplastic)

### Step 4: Metric Calculation
- Compares predictions with ground truth
- Calculates binary and per-type metrics
- Performs cell-level matching for detection metrics

---

## Troubleshooting

### Issue: "FileNotFoundError: images folder not found"

**Solution:** Verify your dataset structure:
```bash
ls ./training_data/crc_codex/256x256_160x160/test/
# Should show: images/ and labels/
```

### Issue: "KeyError: 'inst_map'"

**Solution:** Check your label files contain the correct keys:
```python
import numpy as np
label = np.load("./path/to/label.npy", allow_pickle=True)
label_dict = label.item()
print(label_dict.keys())  # Should show: dict_keys(['inst_map', 'type_map'])
```

### Issue: "ValueError: attempt to get argmax of an empty sequence"

**Solution:** This is already fixed in the current version. If you see this, ensure you're using the latest code.

### Issue: "AssertionError: nuclei_type_map shape mismatch"

**Solution:** This is already fixed. The script now correctly uses `cellvit_model.num_nuclei_classes` for the postprocessor.

---

## Comparison with Other Scripts

| Script | Use Case | Your Dataset? |
|--------|----------|---------------|
| `inference_cellvit_experiment_segmentation.py` | **CRC CODEX** | ✅ **YES!** |
| `inference_cellvit_experiment_consep.py` | CoNSeP benchmark | ❌ No |
| `inference_cellvit_experiment_pannuke.py` | PanNuke (tissue types) | ❌ No |
| `inference_cellvit_experiment_segmentation_old.py` | Generic (backup) | Maybe |

**Use the first one** - it's specifically designed for your CRC CODEX dataset format!

---

## Key Differences from CoNSeP Script

| Feature | CoNSeP | CRC CODEX |
|---------|--------|-----------|
| **File Format** | .json annotations | .npy dictionaries |
| **Label Loading** | JSON parsing | `label.item().get()` |
| **Nuclei Types** | 7 → 4 (merged) | 3 (direct) |
| **Split Name** | Hardcoded "Test" | **Configurable** |
| **Flexibility** | Low | **High** |

---

## Next Steps

1. ✅ Run evaluation on your test set
2. ✅ Review metrics in console and JSON output
3. ✅ Analyze per-type performance
4. ✅ Identify which nuclei types need improvement
5. ✅ Use insights for model refinement

---

## Support

For questions or issues:
1. Check the general evaluation guides in `docs/`
2. Review `TROUBLESHOOTING_COMMON_ERRORS.md`
3. See `UNDERSTANDING_CONSEP_SCRIPT.md` for similar architecture

Your CRC CODEX dataset is now fully supported with custom evaluation infrastructure!
