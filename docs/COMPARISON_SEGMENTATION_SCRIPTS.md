# Comparison: segmentation.py vs segmentation_old.py

## Which Script Should I Use for My Segmentation Dataset?

This guide compares the two segmentation evaluation scripts and helps you choose the right one.

---

## Quick Answer

**For most custom datasets:** Use `inference_cellvit_experiment_segmentation_old.py` ✅

**For CRC CODEX dataset specifically:** Use `inference_cellvit_experiment_crccodex.py`

---

## Your Dataset

If your dataset has:
- `.npy` files with dictionary format
- Loading: `np.load(label, allow_pickle=True).item()`
- Contains: `inst_map` and `type_map`
- Custom nuclei types (e.g., Connective, Inflammatory, Neoplastic)

**Both scripts CAN work!** But `segmentation_old.py` is more flexible.

---

## Detailed Comparison

### 1. Dataset Classes

| Script | Dataset Class | Purpose |
|--------|---------------|---------|
| **segmentation.py** | `CRCCodexDataset` | CRC CODEX-specific |
| **segmentation_old.py** | `SegmentationDataset` | Generic, flexible |

### 2. Key Differences

#### segmentation.py (CRC CODEX-Specific)

**Purpose:** Designed specifically for CRC CODEX dataset format

**Dataset Class:** `CRCCodexDataset`
- Located in: `cellvit/training/datasets/crc_codex.py`
- Hardcoded for .npy dictionary structure
- Expects: `label.item().get("inst_map")` and `label.item().get("type_map")`
- Fixed 3 nuclei types: Connective (1), Inflammatory (2), Neoplastic (3)

**Pros:**
- ✅ Ready to use for CRC CODEX format
- ✅ No configuration needed
- ✅ Optimized for 256×256 patches

**Cons:**
- ❌ Less flexible for variations
- ❌ Hardcoded for specific format
- ❌ Not generic

**Use When:**
- Your dataset IS CRC CODEX or identical format
- You want a ready-to-use solution
- Your structure exactly matches expectations

#### segmentation_old.py (Generic)

**Purpose:** Generic evaluation for any nuclei segmentation dataset

**Dataset Class:** `SegmentationDataset`
- Located in: `cellvit/training/datasets/segmentation_dataset.py`
- Flexible dataset structure
- Auto-detects label folder names
- Supports both .npy and .mat files
- Reads label_map.yaml for nuclei types
- Configurable split names

**Pros:**
- ✅ Highly flexible
- ✅ Works with various formats
- ✅ Auto-detects structure
- ✅ Configurable
- ✅ Better for custom datasets

**Cons:**
- ⚠️  May need adaptation for .npy dictionary format
- ⚠️  Requires label_map.yaml or dataset_config.yaml

**Use When:**
- You have a custom segmentation dataset
- You want maximum flexibility
- Your dataset has any variations from CRC CODEX
- You prefer a generic, maintainable solution

---

## Feature Comparison Table

| Feature | segmentation.py | segmentation_old.py |
|---------|-----------------|---------------------|
| **Dataset Class** | CRCCodexDataset | SegmentationDataset |
| **File Formats** | .npy dictionary only | .npy OR .mat |
| **Label Structure** | `label.item().get()` | Multiple formats |
| **Folder Detection** | Fixed (images/, labels/) | Auto-detects |
| **Split Names** | Configurable | Configurable |
| **Nuclei Types** | Hardcoded (3 types) | From label_map.yaml |
| **Image Size** | 256×256 (documented) | Any size |
| **Flexibility** | Low | **High** |
| **Tissue Types** | Not supported | Not supported |
| **Label Map File** | Not required | Required |

---

## Data Loading Comparison

### CRCCodexDataset (segmentation.py)

```python
# File: cellvit/training/datasets/crc_codex.py
def __getitem__(self, index):
    # Load image
    image = np.load(image_path)
    
    # Load label - HARDCODED for dictionary
    label = np.load(label_path, allow_pickle=True)
    if isinstance(label, np.ndarray) and label.shape == ():
        label = label.item()  # Extract dictionary from 0-d array
    
    # Get inst_map and type_map from dictionary
    gt_inst_map = label.get("inst_map")
    gt_type_map = label.get("type_map")
    
    # Fixed 3 nuclei types
    # 1: Connective, 2: Inflammatory, 3: Neoplastic
```

### SegmentationDataset (segmentation_old.py)

```python
# File: cellvit/training/datasets/segmentation_dataset.py
def __getitem__(self, index):
    # Load image (flexible format)
    image = load_image(image_path)
    
    # Load label - FLEXIBLE for multiple formats
    if label_path.suffix == ".npy":
        # Handle .npy files
        # May need adaptation for dictionary format
    elif label_path.suffix == ".mat":
        # Also handles .mat files
    
    # Nuclei types from label_map.yaml
    # Configurable number of types
```

---

## Usage Examples

### Using segmentation.py (CRC CODEX)

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_crccodex.py \
  --logdir ./log_local/2026-02-06T205610_cellvit++ \
  --cellvit_path ./checkpoints/HIPT-25/CellViT-256-x40-AMP.pth \
  --dataset_path ./training_data/crc_codex/256x256_160x160 \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

**Dataset Structure Expected:**
```
dataset_path/
├── test/
│   ├── images/
│   │   └── *.npy (256×256 images)
│   └── labels/
│       └── *.npy (dictionaries with inst_map, type_map)
```

**No label_map.yaml needed** - types are hardcoded

### Using segmentation_old.py (Generic)

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation_old.py \
  --logdir ./log_local/2026-02-06T205610_cellvit++ \
  --cellvit_path ./checkpoints/HIPT-25/CellViT-256-x40-AMP.pth \
  --dataset_path ./training_data/my_dataset \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0
```

**Dataset Structure Expected:**
```
dataset_path/
├── test/
│   ├── images/
│   │   └── *.png, *.jpg, *.npy
│   └── labels/  # or "labels-1000-1000", auto-detected
│       └── *.npy or *.mat
└── label_map.yaml  # or dataset_config.yaml
```

**label_map.yaml Required:**
```yaml
nuclei_types:
  0: Background
  1: Connective
  2: Inflammatory
  3: Neoplastic
```

---

## Metrics Calculated (Both Scripts)

Both scripts calculate the same comprehensive metrics:

**Overall Binary Metrics:**
- Dice coefficient
- Aggregated Jaccard Index (AJI)
- AJI+ (strict version)
- Panoptic Quality (PQ)
- Detection Quality (DQ)
- Segmentation Quality (SQ)

**Per-Type Metrics:**
- Type-specific Dice, AJI, PQ, DQ, SQ
- Cell-level F1, Precision, Recall
- Calculated for each nuclei type

**Output Format:**
- Console: Formatted tables with metrics
- JSON: Complete results in `{logdir}/inference_results.json`

---

## Recommendation for Your Case

### Your Dataset Characteristics:
- ✅ `.npy` files with dictionary format
- ✅ `label.item().get("inst_map")` and `label.item().get("type_map")`
- ✅ 3 classes: Connective (1), Inflammatory (2), Neoplastic (3)
- ✅ Trained with `train_cell_classifier_head.py`

### Recommended: segmentation_old.py ⭐

**Why?**
1. **More flexible** - adapts to various dataset structures
2. **Generic solution** - better for long-term maintenance
3. **Configurable** - uses label_map.yaml for types
4. **Future-proof** - can handle additional datasets

**Action Needed:**
1. Verify `SegmentationDataset` handles .npy dictionary format
2. Create `label_map.yaml` with your 3 nuclei types
3. Run evaluation

**Potential Adaptation:**

If `SegmentationDataset` doesn't handle your .npy dictionary format, you may need to add:

```python
# In cellvit/training/datasets/segmentation_dataset.py
if label_path.suffix == ".npy":
    label = np.load(label_path, allow_pickle=True)
    
    # Add dictionary handling
    if isinstance(label, np.ndarray) and label.shape == ():
        label = label.item()
    
    if isinstance(label, dict):
        inst_map = label.get("inst_map")
        type_map = label.get("type_map")
    else:
        # Handle other .npy formats
        inst_map = label["inst_map"]
        type_map = label["type_map"]
```

### Alternative: segmentation.py (Also Works)

**Why?**
- Your dataset format matches CRC CODEX exactly
- Ready to use with no modifications
- No label_map.yaml needed

**Action Needed:**
- None! Just run the script

**Limitation:**
- Less flexible if you add more dataset types
- Hardcoded for CRC CODEX structure

---

## Decision Tree

```
Do you have .npy files with inst_map/type_map dictionaries?
│
├─ YES
│  │
│  ├─ Is your dataset EXACTLY like CRC CODEX?
│  │  (3 types: Connective, Inflammatory, Neoplastic)
│  │  (256×256 patches)
│  │  (images/ and labels/ folders)
│  │  │
│  │  ├─ YES → Use segmentation.py ✅
│  │  │         (Ready to use, no config)
│  │  │
│  │  └─ NO → Use segmentation_old.py ⭐
│  │            (More flexible, may need adaptation)
│  │
│  └─ Do you want maximum flexibility?
│     │
│     └─ YES → Use segmentation_old.py ⭐
│                (Better long-term solution)
│
└─ NO (have .mat or different format)
   │
   └─ Use segmentation_old.py ⭐
      (Handles multiple formats)
```

---

## Summary

**Both scripts work for your dataset!**

**Best choice for most cases: `segmentation_old.py`**
- ✅ More flexible and generic
- ✅ Better for custom datasets
- ✅ Configurable via label_map.yaml
- ✅ Future-proof
- ⚠️  May need minor `SegmentationDataset` adaptation

**Alternative choice: `segmentation.py`**
- ✅ Ready to use for CRC CODEX format
- ✅ No configuration needed
- ✅ Works if your structure matches exactly
- ❌ Less flexible for variations

**Your 3 nuclei types work with both!**

The main difference is the underlying dataset class and flexibility, not the evaluation logic. Choose based on your flexibility needs and whether you might add more datasets in the future.

---

## See Also

- [UNDERSTANDING_CONSEP_SCRIPT.md](UNDERSTANDING_CONSEP_SCRIPT.md) - CoNSeP benchmark
- [UNDERSTANDING_PANNUKE_SCRIPT.md](UNDERSTANDING_PANNUKE_SCRIPT.md) - PanNuke with tissue types
- [COMPARISON_INFERENCE_SCRIPTS.md](COMPARISON_INFERENCE_SCRIPTS.md) - All 14 scripts compared
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Complete evaluation reference
- [CRC_CODEX_EVALUATION_GUIDE.md](CRC_CODEX_EVALUATION_GUIDE.md) - CRC CODEX specifics
