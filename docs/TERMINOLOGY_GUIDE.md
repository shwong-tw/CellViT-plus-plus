# CellViT++ Terminology Guide

This guide clarifies the terminology used in CellViT++ to avoid confusion.

## Common Confusion: "Detection" vs "Segmentation"

### ⚠️ Important Clarification

When we say **"Detection Dataset"** or **"Segmentation Dataset"**, we're referring to the **ANNOTATION FORMAT**, not the task!

**Both formats support nuclei type classification!**

---

## Key Terminology

### Dataset Format Types

#### 1. Detection Dataset (DetectionDataset)

**What it means:**
- Annotations are stored as **CSV files** with x, y coordinates
- Each cell is a point location with a class label
- Format: `x, y, label` in CSV files

**What it does NOT mean:**
- ❌ Only cell detection (no classification)
- ❌ Binary classification only

**What it DOES support:**
- ✅ Multi-class nuclei type classification
- ✅ Custom cell types (e.g., Tumor, Stromal, Immune, etc.)
- ✅ Any number of classes

**Example annotation (train_001.csv):**
```csv
x,y,label
150,200,0    # Tumor cell at position (150, 200)
300,450,1    # Stromal cell at position (300, 450)
220,180,2    # Immune cell at position (220, 180)
```

**Use when:**
- You have point annotations (centroid coordinates)
- Your annotations are in CSV format
- You want to train a classifier for nuclei type classification

#### 2. Segmentation Dataset (SegmentationDataset)

**What it means:**
- Annotations are stored as **NumPy files (.npy)** with instance masks
- Each cell is a segmented region (polygon/mask) with a class label
- Format: Dictionary with `inst_map` and `type_map`

**What it does NOT mean:**
- ❌ Only instance segmentation (no classification)
- ❌ Binary segmentation only

**What it DOES support:**
- ✅ Multi-class nuclei type classification
- ✅ Instance segmentation (cell boundaries)
- ✅ Custom cell types
- ✅ Any number of classes

**Example annotation (train_001.npy):**
```python
{
    'inst_map': array([[0, 0, 1, 1, ...],   # Instance IDs (0=background, 1,2,3...=cells)
                       [0, 2, 2, 1, ...],
                       ...]),
    'type_map': array([[0, 0, 1, 1, ...],   # Type IDs (0=background, 1=Tumor, 2=Stromal, etc.)
                       [0, 2, 2, 1, ...],
                       ...])
}
```

**Use when:**
- You have instance segmentation masks
- Your annotations are in NumPy format
- You want to train a classifier AND have precise cell boundaries

---

## Task Types (What You're Actually Doing)

### Task 1: Nuclei Type Classification

**Description:** Assigning a class label to each detected cell (e.g., Tumor, Stromal, Immune, Other)

**Supported by:**
- ✅ DetectionDataset (CSV format)
- ✅ SegmentationDataset (NumPy format)

**Example:** 4-class classifier distinguishing Tumor, Stromal, Immune, and Other cells

### Task 2: Instance Segmentation

**Description:** Detecting cell boundaries and creating individual masks for each cell

**Supported by:**
- ✅ SegmentationDataset (NumPy format) - has precise masks
- ⚠️ DetectionDataset (CSV format) - CellViT generates masks, but annotations are points

### Task 3: Binary Cell Detection

**Description:** Detecting whether cells are present (yes/no) without classification

**Supported by:**
- ✅ Both formats (but typically use binary labels: 0 or 1)

---

## Choosing Your Dataset Format

### Decision Based on Your Annotations

```
What format are your annotations in?

├─ CSV files with (x, y, label) coordinates
│  └─ Use: DetectionDataset
│     Script: train_cell_classifier_head.py with dataset: DetectionDataset
│     Evaluation: inference_cellvit_experiment_detection.py
│
└─ NumPy files with instance masks and type maps
   └─ Use: SegmentationDataset
      Script: train_cell_classifier_head.py with dataset: SegmentationDataset
      Evaluation: inference_cellvit_experiment_pannuke.py (or consep.py, etc.)
```

### NOT Based on Your Task

❌ Don't choose based on:
- Whether you're doing classification
- How many classes you have
- Whether you want per-class metrics

✅ Choose based on:
- What format your annotations are in
- Whether you have point locations or full masks

---

## Common Scenarios

### Scenario 1: "I want to classify nuclei types"

**Question:** Which dataset type should I use?

**Answer:** Depends on your annotation format!
- If you have CSV with coordinates → DetectionDataset
- If you have NumPy with masks → SegmentationDataset

Both support multi-class nuclei type classification!

### Scenario 2: "I have segmentation masks with nuclei types"

**Your data:**
- NumPy files with instance segmentation
- Each instance has a type label (e.g., 1=Tumor, 2=Stromal, etc.)

**What to use:**
- Dataset: SegmentationDataset
- Training: `train_cell_classifier_head.py` with `dataset: SegmentationDataset`
- Evaluation: `inference_cellvit_experiment_pannuke.py` or similar

**NOT "detection" dataset!** Even though you're doing classification, you have segmentation format.

### Scenario 3: "I have point annotations with cell types"

**Your data:**
- CSV files with x, y coordinates
- Each point has a class label (e.g., 0=Tumor, 1=Stromal, etc.)

**What to use:**
- Dataset: DetectionDataset
- Training: `train_cell_classifier_head.py` with `dataset: DetectionDataset`
- Evaluation: `inference_cellvit_experiment_detection.py`

**Yes, "detection" dataset!** Even though you're doing classification, you have point-based format.

### Scenario 4: "I want per-class F1 scores"

**Answer:** Both formats support this!

**DetectionDataset:** 
- Outputs per-class F1, precision, recall in `classifier.per_class` section
- Also outputs pipeline metrics

**SegmentationDataset:**
- Outputs per-class metrics via dataset-specific scripts
- Metrics depend on the benchmark (PQ, DQ, SQ for PanNuke, etc.)

---

## Evaluation Scripts Mapping

### For DetectionDataset (CSV annotations)

**Primary script:** `inference_cellvit_experiment_detection.py`
- Calculates classification metrics (global + per-class)
- Calculates detection quality metrics
- Outputs confusion matrices
- Works with any custom classes
- **Per-class F1 location:** `classifier.per_class[class_name]["f1"]`

**Simplified wrapper:** `inference_cellvit_custom_classifier.py`
- Same functionality, easier to use
- Better error messages and validation

### For SegmentationDataset (NumPy masks)

**Depends on dataset structure:**

**PanNuke-like (with tissue types + nuclei types):**
- Use: `inference_cellvit_experiment_pannuke.py`
- Calculates: PQ, DQ, SQ metrics
- Requires: `dataset_config.yaml` with tissue_types and nuclei_types
- **Per-class F1 location:** `nuclei_metrics_d[nuclei_type]["f1_cell"]`

**CoNSeP-like (nuclei types only):**
- Use: `inference_cellvit_experiment_consep.py`
- Calculates: Segmentation and classification metrics
- Simpler structure than PanNuke
- **Per-class F1 location:** Similar to PanNuke structure

**Other benchmarks:**
- Lizard: `inference_cellvit_experiment_lizard.py`
- MoNuSeg: `inference_cellvit_experiment_monuseg.py`
- etc.

> 📖 **For segmentation datasets:** See [SEGMENTATION_EVALUATION_QUICKSTART.md](SEGMENTATION_EVALUATION_QUICKSTART.md) for detailed guide with examples!

---

## Quick Reference Table

| Your Annotations | Dataset Type | Training Config | Evaluation Script | Per-Class F1 Location | Supports Classification? |
|-----------------|--------------|-----------------|-------------------|-----------------------|------------------------|
| CSV (x,y,label) | DetectionDataset | `dataset: DetectionDataset` | `inference_cellvit_experiment_detection.py` | `classifier.per_class[class]["f1"]` | ✅ Yes (multi-class) |
| NumPy masks + types (with tissue) | SegmentationDataset | `dataset: SegmentationDataset` | `inference_cellvit_experiment_pannuke.py` | `nuclei_metrics_d[type]["f1_cell"]` | ✅ Yes (multi-class) |
| NumPy masks + types (no tissue) | SegmentationDataset | `dataset: SegmentationDataset` | `inference_cellvit_experiment_consep.py` | `nuclei_metrics_d[type]["f1_cell"]` | ✅ Yes (multi-class) |

---

## Summary

**The confusion comes from terminology:**

- **"Detection Dataset"** ≠ "Only does detection"
  - It means "CSV point annotations"
  - But it DOES nuclei type classification!

- **"Segmentation Dataset"** ≠ "Only does segmentation"
  - It means "NumPy mask annotations"
  - But it ALSO does nuclei type classification!

**Key takeaway:**
> Choose your dataset type based on ANNOTATION FORMAT (CSV vs NumPy), not on your TASK (detection vs classification vs segmentation).
> 
> Both formats support multi-class nuclei type classification!

---

## Still Confused?

Ask yourself these questions:

1. **What format are my annotations in?**
   - CSV with coordinates → DetectionDataset
   - NumPy with masks → SegmentationDataset

2. **Do I have full segmentation masks or just points?**
   - Points only → DetectionDataset
   - Full masks → SegmentationDataset

3. **What evaluation metrics do I need?**
   - Classification metrics (F1, precision, recall, AUROC) → Both support this!
   - Segmentation metrics (Dice, PQ, DQ, SQ) → SegmentationDataset is better

If still unclear, check:
- `docs/EVALUATION_GUIDE.md` - Comprehensive evaluation guide
- Your dataset structure in `train_configs/` - Shows which dataset type was used
- Example datasets in `test_database/training_database/` - Examples of both formats
