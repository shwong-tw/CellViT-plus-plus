# Understanding inference_cellvit_experiment_consep.py

This document explains the CoNSeP evaluation script to help you understand when and how to use it.

## ⚠️ Important Notice

**This script is designed specifically for the CoNSeP dataset benchmark.**

CoNSeP = **Co**lorectal **N**uclear **Se**gmentation and **P**henotypes

If you trained a custom classifier using `train_cell_classifier_head.py`, you probably should use:
- `inference_cellvit_experiment_segmentation.py` for generic nuclei-only segmentation datasets
- `inference_cellvit_experiment_detection.py` for detection datasets (CSV annotations)

Only use this script if:
- You're evaluating on the CoNSeP benchmark dataset specifically
- Your dataset has identical structure to CoNSeP

## Script Purpose

This script evaluates CellViT classifiers on the CoNSeP dataset, which has:
1. **Nuclei Type Labels**: 4 cell types (inflammatory, epithelial, spindle-shaped, miscellaneous)
2. **Instance Segmentation Masks**: Stored as .mat or .npy files
3. **No Tissue Types**: Unlike PanNuke, CoNSeP doesn't categorize by tissue
4. **Fixed Split**: Uses predefined "Test" split

## Key Differences from Other Scripts

| Feature | CoNSeP Script | Segmentation Script | PanNuke Script |
|---------|--------------|---------------------|----------------|
| **Inheritance** | Extends CellViTClassifierInferenceExperiment | Extends CellViTClassifierInferenceExperiment | Standalone |
| **Dataset** | CoNSePDataset (hardcoded) | Generic, auto-detects | PanNukeDataset (hardcoded) |
| **Split** | "Test" (hardcoded) | Configurable (--split) | Configurable |
| **Tissue Types** | No | No | Yes |
| **File Format** | .mat or .npy | .npy | .npy |
| **Flexibility** | Low (benchmark-specific) | High (generic) | Low (benchmark-specific) |

## Key Components

### 1. Initialization (Lines 91-110)

```python
class CellViTInfExpCoNSep(CellViTClassifierInferenceExperiment):
    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load CoNSeP Dataset (Used split: Test)"""
```

**What it does:**
- Inherits from base classifier experiment class
- Loads CoNSeP dataset with hardcoded "Test" split
- Returns CoNSePDataset object

**For Custom Datasets:**
- ❌ Cannot change split name (hardcoded to "Test")
- ❌ Cannot use different dataset format
- ✅ Use `inference_cellvit_experiment_segmentation.py` instead (supports --split argument)

### 2. Ground Truth Loading (Lines 112-169)

The script has TWO methods for loading ground truth:

#### Method 1: _load_gt_npy (Lines 112-140)
```python
def _load_gt_npy(
    self, file_name: str, path: Path, subfolder: str = None
) -> Tuple[np.ndarray, np.ndarray]:
```

**What it does:**
- Loads NumPy files (.npy format)
- Extracts instance map and type map
- Converts 0-indexed to 1-indexed for background

**File Structure Expected:**
```
dataset/
└── Test/
    └── Labels/
        └── image_name.npy  # Shape: (H, W, 2) where [:,:,0]=instances, [:,:,1]=types
```

#### Method 2: _load_gt_mat (Lines 142-169)
```python
def _load_gt_mat(
    self, file_name: str, path: Path, subfolder: str = None
) -> Tuple[np.ndarray, np.ndarray]:
```

**What it does:**
- Loads MATLAB files (.mat format) - CoNSeP's original format
- Extracts 'inst_map' and 'type_map' from .mat structure
- Converts 0-indexed to 1-indexed for background

**File Structure Expected:**
```
dataset/
└── Test/
    └── Labels/
        └── image_name.mat  # Contains: {'inst_map': array, 'type_map': array}
```

**For Custom Datasets:**
- Your data must be in one of these two formats
- .npy format: Shape must be (H, W, 2) with [instances, types]
- .mat format: Must have 'inst_map' and 'type_map' keys

### 3. Metrics Calculated (Lines 301-587)

The script calculates comprehensive segmentation and classification metrics:

#### Binary Segmentation Metrics (Per Image)
```python
# Lines 388-402
binary_dice_scores = []
binary_jaccard_scores = []

for inst_map_gt, inst_map_pred in zip(gt_inst_maps, pred_inst_maps):
    dice_score = get_dice_1(inst_map_gt, inst_map_pred)
    binary_dice_scores.append(dice_score)
    
    jacard_score = get_fast_aji(inst_map_gt, inst_map_pred)
    binary_jaccard_scores.append(jacard_score)
```

**Metrics:**
- **Dice Score**: Overlap between predicted and ground truth segmentation
- **Jaccard Index (AJI)**: Aggregated Jaccard Index for instances
- **AJI+**: Extended Jaccard with better handling of over-segmentation

#### Detection Quality Metrics (Per Image)
```python
# Lines 405-434
```

**Metrics:**
- **F1 Detection**: Overall detection performance (detected vs missed vs false positive)
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)

#### Panoptic Quality Metrics (Per Nuclei Type)
```python
# Lines 439-518
pq_metrics = []
for gt_type, pred_type in zip(gt_types, pred_types):
    pq, dq, sq = get_pq(gt_type, pred_type)
    pq_metrics.append((pq, dq, sq))
```

**Metrics:**
- **PQ (Panoptic Quality)**: Combined segmentation + classification quality
- **DQ (Detection Quality)**: How well instances are detected/matched
- **SQ (Segmentation Quality)**: IoU of matched instances

These metrics are calculated:
- **Per nuclei type**: Separate scores for inflammatory, epithelial, spindle-shaped, miscellaneous
- **Binary PQ (bPQ)**: Ignores cell types, just checks if cells detected
- **Multi-class PQ (mPQ)**: Requires correct type prediction

#### Classification Metrics (Global)
```python
# Lines 214-253 in _get_global_classifier_scores()
```

**Metrics:**
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Correct predictions / all predictions
- **Recall**: Correct predictions / all ground truth
- **Accuracy**: Overall correctness
- **AUROC**: Area under ROC curve
- **Average Precision (AP)**: Area under precision-recall curve

### 4. Output Format (Lines 564-587)

Results are saved to `{outdir}/inference_results.json`:

```json
{
  "dataset": {
    "Binary-Cell-Dice-Mean": 0.8234,
    "Binary-Cell-Jacard-Mean": 0.7123,
    "Binary-Cell-Jacard-Plus-Mean": 0.7456,
    "bPQ": 0.6534,
    "mPQ": 0.5987,
    "f1_detection": 0.8156,
    "precision_detection": 0.8234,
    "recall_detection": 0.8078
  },
  "nuclei_binary_metrics": {
    "Dice": 0.8234,
    "Jacard": 0.7123,
    "Jacard-Plus": 0.7456
  },
  "nuclei_metrics_pq": {
    "Background": 0.0,
    "Inflammatory": 0.6234,
    "Epithelial": 0.7123,
    "Spindle-shaped": 0.5678,
    "Miscellaneous": 0.4321
  },
  "nuclei_metrics_d": {
    "Inflammatory": {
      "f1_cell": 0.8156,
      "prec_cell": 0.8234,
      "rec_cell": 0.8078,
      "PQ": 0.6234,
      "DQ": 0.7890,
      "SQ": 0.7901
    },
    // ... other types
  }
}
```

## Comparison: CoNSeP vs Segmentation Script

The new `inference_cellvit_experiment_segmentation.py` we created is more flexible:

| Aspect | CoNSeP Script | Segmentation Script (NEW) |
|--------|--------------|---------------------------|
| **Dataset Type** | CoNSePDataset only | Any segmentation dataset |
| **Split Name** | "Test" (hardcoded) | Configurable via --split |
| **Label Format** | .mat or .npy | .npy (auto-detected) |
| **Tissue Types** | Not supported | Not required |
| **CLI Arguments** | Limited | Comprehensive (--split, --checkpoint_name, etc.) |
| **Error Handling** | Basic | Enhanced with validation |
| **Documentation** | This file | HOW_TO_RUN_SEGMENTATION_EVALUATION.md |

**When to use CoNSeP script:**
- Evaluating on official CoNSeP benchmark
- Comparing results with published papers
- Need .mat file support

**When to use Segmentation script:**
- Custom datasets with nuclei-only segmentation
- Flexible evaluation needs
- Better error messages and validation

## Usage Example

```bash
# Basic usage
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_consep.py \
  --logdir ./logs/my_consep_training \
  --cellvit_path ./models/CellViT-256-x40.pth \
  --dataset_path ./data/consep \
  --gpu 0

# The script will:
# 1. Load CoNSePDataset with "Test" split
# 2. Load your trained classifier from logdir/checkpoints/model_best.pth
# 3. Run inference on all test images
# 4. Calculate all metrics
# 5. Save results to logdir/results/inference_results.json
```

## What Gets Calculated

For each test image:
1. **Segmentation Quality**: How well cells are segmented (Dice, AJI)
2. **Detection Quality**: How many cells are found (F1, Precision, Recall)
3. **Classification Quality**: How well types are predicted (per-type PQ, DQ, SQ)

Aggregated across dataset:
1. **Binary metrics**: Average Dice, Jaccard (ignoring types)
2. **Detection metrics**: Overall F1, Precision, Recall
3. **Panoptic metrics**: PQ, DQ, SQ per nuclei type
4. **Classification metrics**: Global F1, AUROC, AP

## Common Issues

### Issue 1: "Test" split not found
```
Error: Split 'Test' not found in dataset
```

**Cause:** Your dataset doesn't have a folder named "Test"

**Solution:** 
- Use `inference_cellvit_experiment_segmentation.py` with --split argument
- Or rename your folder to "Test"

### Issue 2: Wrong label format
```
Error: Cannot load .npy file, expected shape (H, W, 2)
```

**Cause:** Your .npy files don't have the expected structure

**Solution:**
- Ensure .npy files have shape (H, W, 2)
- Index 0: instance map, Index 1: type map
- Or use .mat format with 'inst_map' and 'type_map' keys

### Issue 3: Nuclei type mismatch
```
KeyError: 'Inflammatory'
```

**Cause:** Your dataset has different nuclei types than CoNSeP

**Solution:**
- Use `inference_cellvit_experiment_segmentation.py` (supports custom types)
- Or modify your label_map.yaml to match CoNSeP types

## Key Takeaway

**For most users with custom datasets:**
- ❌ Do NOT use this script
- ✅ Use `inference_cellvit_experiment_segmentation.py` instead

**Only use this script if:**
- ✅ Evaluating on official CoNSeP benchmark
- ✅ Your dataset has identical structure to CoNSeP
- ✅ You need .mat file format support

## See Also

- `docs/EVALUATION_GUIDE.md` - Comprehensive evaluation guide
- `docs/SEGMENTATION_EVALUATION_QUICKSTART.md` - Quick start for segmentation
- `docs/UNDERSTANDING_PANNUKE_SCRIPT.md` - Similar guide for PanNuke
- `docs/COMPARISON_INFERENCE_SCRIPTS.md` - Compare all inference scripts
- `docs/HOW_TO_RUN_SEGMENTATION_EVALUATION.md` - Step-by-step usage guide
