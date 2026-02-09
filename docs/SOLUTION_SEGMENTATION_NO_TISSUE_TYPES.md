# SOLUTION: Segmentation Dataset with Only Nuclei Classes (No Tissue Types)

## Your Situation

- ✅ Segmentation annotation dataset (NumPy masks)
- ✅ Nuclei class labels (e.g., Tumor, Non-Tumor, Inflammatory, etc.)
- ❌ **NO tissue type labels**
- ✅ Completed retraining
- ❓ Need per-class F1 scores

## The Solution: Use the New Generic Script! ⭐

**GOOD NEWS:** We now have a dedicated script for nuclei-only segmentation datasets!

**USE THIS SCRIPT:** `inference_cellvit_experiment_nuclei_segmentation.py`

**WHY:** This script is specifically designed for segmentation datasets with only nuclei classes (no tissue types required).

---

## Quick Start

### Step 1: Verify Your Dataset Structure

Your dataset should look like this:

```
your_dataset/
├── test/                 # Or "val", "Test", etc.
│   ├── images/
│   │   └── *.png, *.jpg
│   └── labels/           # Or "labels-1000-1000", etc.
│       └── *.npy or *.mat (containing inst_map and type_map)
└── label_map.yaml        # Nuclei type definitions
```

### Step 2: Check Your label_map.yaml

This file defines your nuclei classes:

```yaml
1: "Tumor"
2: "Non-Tumor"
3: "Inflammatory"
4: "Stromal"
# Add all your nuclei classes here (0 is reserved for Background)
```

**Note:** You can also use `dataset_config.yaml` with a `nuclei_types` section - the script auto-detects both formats.

### Step 3: Run the Evaluation

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_nuclei_segmentation.py \
  --logdir /path/to/your/training/run \
  --cellvit_path /path/to/cellvit/model.pth \
  --dataset_path /path/to/your/dataset \
  --checkpoint_name model_best.pth \
  --split test \
  --gpu 0
```

**Replace these values:**
- `/path/to/your/training/run` → Your training run directory (e.g., `./logs/my_training_2024_01_15`)
- `/path/to/cellvit/model.pth` → Path to CellViT backbone checkpoint
- `/path/to/your/dataset` → Path to your dataset folder
- `--split test` → Your test split name (test, val, Test, etc.)

### Step 4: Get Per-Class F1 Scores

**Console Output:**

```
******************** Nuclei Detection Metrics ********************
Nuclei Type          Precision      Recall          F1
----------------------------------------------------------------
Tumor                   0.872        0.845       0.858    ← Per-class F1
Non-Tumor               0.823        0.798       0.810    ← Per-class F1
Inflammatory            0.791        0.774       0.782    ← Per-class F1
Stromal                 0.745        0.712       0.728    ← Per-class F1
----------------------------------------------------------------
Average                 0.808        0.782       0.795    ← Average F1
```

**JSON Output:**

Results saved to `{logdir}/test_results/inference_results.json`:

```json
{
  "nuclei_metrics_d": {
    "Tumor": {
      "f1_cell": 0.858,
      "prec_cell": 0.872,
      "rec_cell": 0.845,
      "pq": 0.654,
      "dq": 0.789,
      "sq": 0.829
    },
    "Non-Tumor": {
      "f1_cell": 0.810,
      ...
    },
    ...
  },
  "dataset": {
    "f1_detection": 0.795,    // Average F1
    "prec_detection": 0.808,
    "rec_detection": 0.782
  }
}
```

---

## Advanced Usage

### Different Ground Truth Format

If your labels are in .mat format instead of .npy:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_nuclei_segmentation.py \
  --logdir /path/to/your/training/run \
  --cellvit_path /path/to/cellvit/model.pth \
  --dataset_path /path/to/your/dataset \
  --checkpoint_name model_best.pth \
  --split test \
  --gt_format mat \
  --gpu 0
```

### Different Label Map File

If your label map file has a different name:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_nuclei_segmentation.py \
  --logdir /path/to/your/training/run \
  --cellvit_path /path/to/cellvit/model.pth \
  --dataset_path /path/to/your/dataset \
  --checkpoint_name model_best.pth \
  --label_map_file my_custom_labels.yaml \
  --gpu 0
```

### With Stain Normalization

If you used stain normalization during training:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_nuclei_segmentation.py \
  --logdir /path/to/your/training/run \
  --cellvit_path /path/to/cellvit/model.pth \
  --dataset_path /path/to/your/dataset \
  --checkpoint_name model_best.pth \
  --normalize_stains \
  --gpu 0
```

---

## Complete Example

### Your Dataset

```
/data/my_nuclei_dataset/
├── test/
│   ├── images/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── labels/
│       ├── img_001.npy
│       ├── img_002.npy
│       └── ...
└── label_map.yaml
```

### Your label_map.yaml

```yaml
1: "Epithelial"
2: "Lymphocyte"
3: "Neutrophil"
4: "Macrophage"
```

### Command to Run

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_nuclei_segmentation.py \
  --logdir ./logs/my_epithelial_training_2024_01_15 \
  --cellvit_path ./models/cellvit_sam_h.pth \
  --dataset_path /data/my_nuclei_dataset \
  --checkpoint_name model_best.pth \
  --split test \
  --gpu 0
```

### Output

The script will create:
- `./logs/my_epithelial_training_2024_01_15/test_results/inference_results.json` - All metrics
- `./logs/my_epithelial_training_2024_01_15/test_results/cell_predictions/` - Per-image predictions
- `./logs/my_epithelial_training_2024_01_15/test_results/confusion_matrix.png` - Confusion matrix

---

## Key Advantages

✅ **No tissue types needed** - Works with nuclei classes only  
✅ **No dummy data** - No need to modify your training dataset  
✅ **Generic and flexible** - Works with any segmentation dataset structure  
✅ **Per-class F1 scores** - Automatic calculation for all nuclei types  
✅ **Multiple metrics** - F1, Precision, Recall, PQ, DQ, SQ for each class  
✅ **Easy to use** - Minimal configuration required  

---

## Comparison: Old vs New Approach

### ❌ Old Approach (Dummy Tissue Type)
- Required creating dummy tissue type in config
- Might confuse evaluation logic
- Not ideal for production use
- Workaround, not a real solution

### ✅ New Approach (Generic Script)
- Designed specifically for nuclei-only datasets
- Clean and straightforward
- No dummy data needed
- Production-ready

---

## Troubleshooting

### Error: "label_map.yaml not found"

**Cause:** The script can't find your label map file.

**Solution:** Either create `label_map.yaml` in your dataset folder, or specify the path:

```bash
--label_map_file /path/to/your/label_map.yaml
```

### Error: "labels folder not found"

**Cause:** The script can't find the labels folder in your split directory.

**Solution:** Make sure your dataset structure matches:
```
dataset_path/
└── {split}/
    └── labels/  # Or "labels-1000-1000", "Labels", etc.
```

The script auto-detects common folder names. If yours is different, it will show an error with the expected path.

### Error: "Ground truth file not found"

**Cause:** Your .npy or .mat files don't match the image names.

**Solution:** Ensure that for each `images/img_001.png`, there's a corresponding `labels/img_001.npy` (or `.mat`).

### Different Results Than Expected

**Check:**
1. Are you using the same model checkpoint you trained?
2. Is your dataset the same as during training?
3. Did you use stain normalization during training? (Add `--normalize_stains` if yes)

---

## Summary

**For segmentation datasets WITHOUT tissue types:**

1. **Use:** `inference_cellvit_experiment_nuclei_segmentation.py`
2. **Requires:** Only nuclei type labels (no tissue types needed)
3. **Provides:** Per-class F1 scores for each nuclei type
4. **Benefits:** Clean, generic, production-ready

**No need for dummy tissue types or data modifications!**

This is the recommended solution for all nuclei-only segmentation datasets.
