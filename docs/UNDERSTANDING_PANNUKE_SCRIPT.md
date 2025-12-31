# Understanding inference_cellvit_experiment_pannuke.py

This document explains the PanNuke evaluation script to help you understand what needs to be modified for custom classifiers.

## ⚠️ Important Notice

**This script is designed specifically for the PanNuke dataset structure.**

If you trained a custom classifier using `train_cell_classifier_head.py`, you should use:
- `inference_cellvit_experiment_detection.py` for detection datasets (CSV annotations)
- `inference_cellvit_custom_classifier.py` for a simplified interface

Only use or modify this script if your dataset has the same structure as PanNuke.

## Script Purpose

This script evaluates CellViT models trained on the PanNuke dataset, which has:
1. **Tissue Types**: Different tissue categories (e.g., breast, colon, prostate)
2. **Nuclei Types**: Different cell types within tissues (e.g., neoplastic, inflammatory, connective)
3. **Segmentation Masks**: Instance segmentation with type labels

## Key Components

### 1. Dataset Requirements (Lines 108-122)

The script expects a `dataset_config.yaml` file in your dataset directory:

```yaml
tissue_types:
  Adrenal_gland: 0
  Bile-duct: 1
  Bladder: 2
  # ... more tissue types

nuclei_types:
  Background: 0
  Neoplastic: 1
  Inflammatory: 2
  Connective: 3
  Dead: 4
  Epithelial: 5
```

**For Custom Classifiers:**
- If you only have nuclei types (no tissue types), you need to modify lines 108-122
- If your labels are different, update the dataset_config.yaml accordingly

### 2. Model Loading (Lines 145-214)

The script loads a full CellViT segmentation model, not just a classifier head.

**For Custom Classifiers:**
- If you trained only a classification head, use `inference_cellvit_experiment_detection.py` instead
- That script loads the base CellViT model + your classifier head separately

### 3. Dataset Loading (Lines 263-278)

The script uses `select_dataset()` which expects specific dataset types like:
- PanNukeDataset
- CoNSePDataset
- LizardDataset

**For Custom Classifiers:**
- Detection datasets use DetectionDataset
- This is already handled correctly in `inference_cellvit_experiment_detection.py`

### 4. Metrics Calculated (Lines 290-573)

The script calculates several types of metrics:

#### Binary Metrics (per image)
- **Dice Score**: Overlap between predicted and ground truth cells
- **Jaccard Index**: Intersection over union for cells

#### Segmentation Quality Metrics
- **PQ (Panoptic Quality)**: Overall segmentation + classification quality
- **DQ (Detection Quality)**: How well instances are detected
- **SQ (Segmentation Quality)**: How well instances are segmented

#### Tissue-Specific Metrics (Lines 446-466)
- Performance broken down by tissue type
- Only relevant for multi-tissue datasets like PanNuke

#### Nuclei Type Metrics (Lines 469-497)
- Performance per cell class
- Detection scores (F1, precision, recall) for each type

**For Custom Classifiers:**
- Detection script provides similar per-class metrics
- But doesn't require tissue types
- Focuses on classification performance (AUROC, F1) rather than segmentation (PQ, DQ, SQ)

### 5. Output Format (Lines 556-573)

Results are saved to `inference_results.json`:

```json
{
  "dataset": {
    "Binary-Cell-Dice-Mean": 0.85,
    "Binary-Cell-Jacard-Mean": 0.75,
    "bPQ": 0.65,
    "mPQ": 0.60,
    "f1_detection": 0.80
  },
  "tissue_metrics": {
    "breast": {"Dice": 0.87, "mPQ": 0.62},
    ...
  },
  "nuclei_metrics_pq": {
    "Neoplastic": 0.70,
    "Inflammatory": 0.65,
    ...
  }
}
```

**For Custom Classifiers:**
- Detection script outputs AUROC, F1, Precision, Recall instead
- Includes confusion matrices
- Focuses on classification rather than segmentation quality

## What You Would Need to Modify for Custom Classes

If you must use this script with custom classes, here's what to change:

### Option 1: Simple Nuclei Classification (No Tissue Types)

1. **Create dataset_config.yaml** in your dataset:
```yaml
tissue_types:
  Generic: 0  # Single tissue type

nuclei_types:
  Background: 0
  YourClass1: 1
  YourClass2: 2
  YourClass3: 3
  # Add all your classes
```

2. **Modify your dataset** to return tissue type as "Generic" for all images

3. **Update your training config** to match the nuclei_types

### Option 2: Multiple Tissue Types

1. **Create dataset_config.yaml** with your tissue and nuclei types

2. **Ensure your dataset** provides:
   - Instance segmentation masks
   - Nuclei type labels for each instance
   - Tissue type for each image

3. **Match the dataset structure** to what PanNuke expects (see example datasets)

## Recommendation

**Do NOT modify this script if you:**
- Trained with DetectionDataset (CSV annotations with x, y coordinates)
- Only have classification labels (no instance segmentation masks)
- Don't have tissue types

**Instead, use:**
- `inference_cellvit_experiment_detection.py` - Full featured for detection data
- `inference_cellvit_custom_classifier.py` - Simplified wrapper with better UX

**Only modify this script if:**
- Your dataset has PanNuke-like structure (tissue types + nuclei segmentation)
- You need PQ/DQ/SQ metrics (segmentation quality metrics)
- You trained a full segmentation model (not just classifier head)

## Comparison: PanNuke Script vs Detection Script

| Feature | inference_cellvit_experiment_pannuke.py | inference_cellvit_experiment_detection.py |
|---------|-----------|--------------|
| Input Data | Segmentation masks | CSV coordinates |
| Tissue Types | Required | Not used |
| Metrics | PQ, DQ, SQ, Dice | AUROC, F1, Precision, Recall |
| Model Type | Full segmentation | Segmentation + classifier head |
| Use Case | Benchmark datasets | Custom classifiers |
| Complexity | High | Medium |
| Output | Segmentation quality | Classification quality |

## Getting Help

If you're unsure which script to use:
1. Check your training config: `data.dataset` field
   - `DetectionDataset` → Use `inference_cellvit_experiment_detection.py`
   - `PanNukeDataset` → Use `inference_cellvit_experiment_pannuke.py`
2. Check your labels:
   - CSV files with coordinates → Use `inference_cellvit_experiment_detection.py`
   - NumPy instance masks → Use `inference_cellvit_experiment_pannuke.py` or `inference_cellvit_experiment_consep.py`
3. See `docs/EVALUATION_GUIDE.md` for detailed guidance

## Key Takeaway

**For most users training custom classifiers: Do NOT use this script.**

The PanNuke script is highly specialized for segmentation benchmarks. For custom cell classification tasks, use the detection evaluation script which is designed to work with any custom class labels.
