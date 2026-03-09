# Understanding CellViT++ Segmentation Evaluation Scripts

This guide explains the two segmentation evaluation scripts and helps you choose which one to use.

---

## 📋 Quick Overview

| Script | Purpose | Best For | Flexibility |
|--------|---------|----------|-------------|
| **`inference_cellvit_experiment_segmentation.py`** | Generic nuclei segmentation | Custom datasets, flexible needs | **High** ⭐ |
| **`inference_cellvit_experiment_crccodex.py`** | CRC CODEX benchmark | CRC CODEX dataset specifically | Low |

---

## 🎯 Quick Decision

**Choose based on your dataset:**

- ✅ **Use `segmentation.py`** → For most cases (custom datasets, flexibility needed)
- ✅ **Use `crccodex.py`** → Only if your dataset IS CRC CODEX or identical format

---

## 1. Generic Segmentation Script ⭐

### `inference_cellvit_experiment_segmentation.py`

**Purpose:** Flexible evaluation for any nuclei-only segmentation dataset

### Key Features

✅ **Generic `SegmentationDataset` class**
- Works with various data structures
- Configurable nuclei types
- Auto-detects folder layouts

✅ **Multiple file formats**
- Supports both `.npy` and `.mat` files
- Flexible label loading
- Handles different structures

✅ **Configurable via label_map.yaml**
- Define your nuclei types
- Set class names
- Background class (0) automatic

✅ **Highly flexible**
- Works with custom datasets
- Adaptable to various formats
- Future-proof design

### Dataset Requirements

**Structure:**
```
dataset_path/
├── {split}/              # e.g., test, val, train
│   ├── images/
│   │   └── *.npy or *.png, *.jpg
│   └── labels/
│       └── *.npy or *.mat
└── label_map.yaml        # Required configuration file
```

**label_map.yaml Example:**
```yaml
nuclei_types:
  0: Background
  1: Tumor
  2: Stromal
  3: Immune
```

### When to Use

✅ Custom nuclei segmentation datasets
✅ Need flexibility in data structure
✅ Want to configure nuclei types
✅ Planning to use multiple datasets
✅ Not using CRC CODEX specifically

### Usage Example

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir ./log_local/my_training_run \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --dataset_path ./data/my_custom_dataset \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

---

## 2. CRC CODEX Script

### `inference_cellvit_experiment_crccodex.py`

**Purpose:** CRC CODEX benchmark evaluation (dataset-specific)

### Key Features

✅ **CRC CODEX-specific `CRCCodexDataset`**
- Designed for CRC CODEX format
- Hardcoded structure
- Optimized for this dataset

✅ **Fixed configuration**
- 3 nuclei types (Connective, Inflammatory, Neoplastic)
- 256×256 pixel patches
- No configuration file needed

✅ **.npy dictionary format**
- `np.load(label, allow_pickle=True).item()`
- `inst_map` and `type_map` keys
- Specific loading pattern

✅ **Ready to use**
- No setup needed for CRC CODEX
- Works out of the box
- No label_map.yaml required

### Dataset Requirements

**Structure:**
```
dataset_path/
└── {split}/              # e.g., test, val, train
    ├── images/
    │   └── *.npy         # 256×256 pixel patches
    └── labels/
        └── *.npy         # Dictionary with inst_map, type_map
```

**Label Format:**
```python
label = np.load(label_path, allow_pickle=True)
label_dict = label.item()  # Get dictionary
inst_map = label_dict.get("inst_map")  # (256, 256)
type_map = label_dict.get("type_map")  # (256, 256)
```

**Nuclei Types (Hardcoded):**
- 0: Background
- 1: Connective
- 2: Inflammatory
- 3: Neoplastic

### When to Use

✅ Your dataset IS CRC CODEX
✅ Dataset matches CRC CODEX format exactly
✅ You trained specifically on CRC CODEX
✅ Want ready-to-use solution with no config

### Usage Example

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_crccodex.py \
  --logdir ./log_local/crc_training_run \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --dataset_path ./data/crc_codex/256x256_160x160 \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

---

## 📊 Detailed Comparison

### Feature Comparison

| Feature | segmentation.py (Generic) | crccodex.py (Specific) |
|---------|---------------------------|------------------------|
| **Dataset Class** | SegmentationDataset | CRCCodexDataset |
| **Flexibility** | **High** ⭐ | Low |
| **File Formats** | .npy OR .mat | .npy dict only |
| **Nuclei Types** | Configurable (yaml) | Hardcoded (3 types) |
| **Label Map** | Required | Not needed |
| **Image Size** | Any size | 256×256 patches |
| **Folder Detection** | Auto-detects | Fixed structure |
| **Custom Datasets** | ✅ **Excellent** | ❌ Not suitable |
| **CRC CODEX** | ✅ Works (with config) | ✅ **Optimized** |
| **Future-Proof** | ✅ Yes | ⚠️ Dataset-specific |

### Data Loading Comparison

**Generic (segmentation.py):**
```python
# Flexible loading via SegmentationDataset
if label_path.suffix == ".npy":
    # Handles various .npy structures
elif label_path.suffix == ".mat":
    # Also handles .mat files
# Nuclei types from label_map.yaml
```

**CRC CODEX (crccodex.py):**
```python
# Specific loading via CRCCodexDataset
label = np.load(label_path, allow_pickle=True)
label_dict = label.item()
inst_map = label_dict.get("inst_map")
type_map = label_dict.get("type_map")
# Nuclei types hardcoded: Connective, Inflammatory, Neoplastic
```

---

## 🌳 Decision Tree

```
START: Need to evaluate nuclei segmentation?
│
├─ Is your dataset CRC CODEX or EXACTLY the same format?
│  │
│  ├─ YES, it's CRC CODEX
│  │  └─ Use: crccodex.py ✅
│  │     (Ready to use, no config needed)
│  │
│  └─ NO, it's a custom dataset
│     └─ Use: segmentation.py ⭐
│        (Flexible, works with any format)
│
├─ Do you need flexibility for multiple datasets?
│  │
│  └─ YES
│     └─ Use: segmentation.py ⭐
│        (Configure via label_map.yaml)
│
└─ Are your nuclei types different from CRC CODEX?
   │
   └─ YES
      └─ Use: segmentation.py ⭐
         (Define custom types in yaml)
```

**Recommendation: Use `segmentation.py` for most cases! ⭐**

---

## 📝 Complete Usage Examples

### Example 1: Generic Segmentation (Recommended)

**Dataset:** Custom nuclei dataset with 4 types

**Setup:**
```bash
# Create label_map.yaml
cat > my_dataset/label_map.yaml << EOF
nuclei_types:
  0: Background
  1: Epithelial
  2: Stromal
  3: Immune
  4: Necrotic
EOF
```

**Run:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir ./logs/my_model \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --dataset_path ./data/my_dataset \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0
```

### Example 2: CRC CODEX Specific

**Dataset:** CRC CODEX with 3 types (Connective, Inflammatory, Neoplastic)

**Run:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_crccodex.py \
  --logdir ./logs/crc_model \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --dataset_path ./data/crc_codex/256x256_160x160 \
  --split test \
  --checkpoint_name model_best.pth \
  --gpu 0
```

---

## 📈 Metrics Calculated

**Both scripts calculate the same comprehensive metrics:**

### Overall Binary Metrics
- **Dice coefficient** - Overlap measure
- **AJI** (Aggregated Jaccard Index) - Instance matching
- **AJI+** - Strict instance matching
- **PQ** (Panoptic Quality) - Overall quality
- **DQ** (Detection Quality) - Detection accuracy
- **SQ** (Segmentation Quality) - Segmentation accuracy

### Per-Type Metrics (for each nuclei type)
- Type-specific **Dice, AJI, PQ, DQ, SQ**
- Cell-level **F1, Precision, Recall**
- Confusion matrices

### Output Format
- **Console:** Formatted tables with all metrics
- **JSON:** `{logdir}/inference_results.json` with complete results

---

## 🔧 Common Arguments

Both scripts share these command-line arguments:

| Argument | Description | Example | Required |
|----------|-------------|---------|----------|
| `--logdir` | Training output directory | `./logs/my_run` | ✅ Yes |
| `--cellvit_path` | CellViT base model path | `./checkpoints/CellViT.pth` | ✅ Yes |
| `--dataset_path` | Dataset root folder | `./data/my_dataset` | ✅ Yes |
| `--split` | Dataset split to evaluate | `test` (default: `"test"`) | No |
| `--checkpoint_name` | Classifier checkpoint name | `model_best.pth` (default) | No |
| `--gpu` | GPU device ID | `0` (default: `0`) | No |
| `--magnification` | Magnification level | `40` (default: `40`) | No |

---

## ⚠️ Common Issues

### Issue 1: "label_map.yaml not found"

**Solution for segmentation.py:**
```bash
# Create label_map.yaml in dataset root
cat > dataset_path/label_map.yaml << EOF
nuclei_types:
  0: Background
  1: Type1
  2: Type2
EOF
```

**Or use crccodex.py** if your dataset matches CRC CODEX format.

### Issue 2: "Wrong number of nuclei types"

**For segmentation.py:**
- Update label_map.yaml to match your trained classifier
- Ensure types match training configuration

**For crccodex.py:**
- Only works with 3 nuclei types (Connective, Inflammatory, Neoplastic)
- If you have different types, use segmentation.py

### Issue 3: "Dataset structure not found"

**Check folder structure:**
```bash
dataset_path/
├── test/
│   ├── images/
│   └── labels/
└── label_map.yaml  # For segmentation.py only
```

---

## 🔗 See Also

**Related Documentation:**
- [CRC_CODEX_EVALUATION_GUIDE.md](CRC_CODEX_EVALUATION_GUIDE.md) - Deep dive into CRC CODEX script
- [COMPARISON_SEGMENTATION_SCRIPTS.md](COMPARISON_SEGMENTATION_SCRIPTS.md) - Detailed technical comparison
- [HOW_TO_RUN_SEGMENTATION_EVALUATION.md](HOW_TO_RUN_SEGMENTATION_EVALUATION.md) - Complete usage guide
- [SEGMENTATION_EVALUATION_QUICKSTART.md](SEGMENTATION_EVALUATION_QUICKSTART.md) - Quick start
- [UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md](UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md) - Architecture explanation

---

## 💡 Key Takeaways

### For Most Users: Use `segmentation.py` ⭐

**Reasons:**
1. ✅ Works with any nuclei segmentation dataset
2. ✅ Configurable via label_map.yaml
3. ✅ Flexible and future-proof
4. ✅ Handles various file formats
5. ✅ Recommended for custom datasets

### For CRC CODEX Users: Use `crccodex.py`

**Reasons:**
1. ✅ Optimized for CRC CODEX format
2. ✅ No configuration needed
3. ✅ Ready to use out of the box
4. ✅ Hardcoded 3 nuclei types
5. ✅ Perfect if dataset matches exactly

### Bottom Line

**Unless you're specifically using CRC CODEX dataset, use `segmentation.py`!**

It's more flexible, works with any dataset, and gives you control over your nuclei types through configuration.

---

## 📚 Summary

| Aspect | segmentation.py | crccodex.py |
|--------|-----------------|-------------|
| **Use For** | **Custom datasets** ⭐ | CRC CODEX only |
| **Flexibility** | **High** | Low |
| **Configuration** | label_map.yaml | None needed |
| **Nuclei Types** | **Configurable** | Fixed (3 types) |
| **Recommendation** | **Primary choice** ⭐ | Specific use case |

**Choose wisely based on your dataset type and flexibility needs!**
