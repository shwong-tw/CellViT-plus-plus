# Quick Guide: Evaluating Segmentation Datasets with Per-Class F1 Scores

## Your Situation

✅ You have: Segmentation annotation dataset (NumPy masks with instance + type labels)  
✅ You completed: Retraining with your custom dataset  
✅ You need: Evaluation script that provides **per-class F1 scores** + average F1

## Which Script to Use?

### Option 1: Dataset with Tissue Types (PanNuke-style)

**Use:** `inference_cellvit_experiment_pannuke.py`

**When:**
- Your dataset has tissue types (e.g., breast, colon, prostate, etc.)
- You have a `dataset_config.yaml` with both `tissue_types` and `nuclei_types`

**Command:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir /path/to/your/training/run \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

### Option 2: Dataset without Tissue Types (CoNSeP-style)

**Use:** `inference_cellvit_experiment_consep.py`

**When:**
- Your dataset structure matches the **CoNSeP benchmark dataset** exactly
- You have only nuclei types (no tissue types)
- You're evaluating on the CoNSeP dataset itself

**⚠️ IMPORTANT:** This script is specifically for the CoNSeP benchmark dataset. 
If you have a custom segmentation dataset without tissue types, you may need to:
- Create a custom evaluation script based on this one
- Or adapt your dataset to match CoNSeP structure

**Command:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_consep.py \
  --logdir /path/to/your/training/run \
  --dataset_path /path/to/consep/dataset \
  --cellvit_path /path/to/cellvit/model \
  --checkpoint_name model_best.pth \
  --gpu 0
```

---

## Output: Per-Class F1 Scores

### ✅ Both scripts already provide per-class F1 scores!

The evaluation scripts calculate and display:

### Console Output

```
******************** Nuclei Detection Metrics ********************
Nuclei Type          Precision      Recall          F1
----------------------------------------------------------------
Neoplastic              0.872        0.845       0.858
Inflammatory            0.823        0.798       0.810
Connective              0.791        0.774       0.782
Dead                    0.745        0.712       0.728
Epithelial              0.834        0.821       0.827
```

### JSON Output (inference_results.json)

```json
{
  "dataset": {
    "Binary-Cell-Dice-Mean": 0.851,
    "bPQ": 0.632,
    "mPQ": 0.598,
    "f1_detection": 0.803
  },
  "nuclei_metrics_d": {
    "Neoplastic": {
      "f1_cell": 0.858,
      "prec_cell": 0.872,
      "rec_cell": 0.845
    },
    "Inflammatory": {
      "f1_cell": 0.810,
      "prec_cell": 0.823,
      "rec_cell": 0.798
    },
    "Connective": {
      "f1_cell": 0.782,
      "prec_cell": 0.791,
      "rec_cell": 0.774
    },
    "Dead": {
      "f1_cell": 0.728,
      "prec_cell": 0.745,
      "rec_cell": 0.712
    },
    "Epithelial": {
      "f1_cell": 0.827,
      "prec_cell": 0.834,
      "rec_cell": 0.821
    }
  },
  "nuclei_metrics_pq": {
    "Neoplastic": 0.702,
    "Inflammatory": 0.654,
    "Connective": 0.612,
    "Dead": 0.534,
    "Epithelial": 0.687
  }
}
```

---

## What Metrics Are Provided?

### Global Metrics (Average Across All Classes)

- **Binary-Cell-Dice-Mean**: Average Dice score for cell detection
- **bPQ**: Binary Panoptic Quality (segmentation + detection)
- **mPQ**: Mean Panoptic Quality (averaged across nuclei types)
- **f1_detection**: Overall F1 for cell detection

### Per-Class Metrics (For Each Nuclei Type)

**`nuclei_metrics_d`** - Detection & Classification metrics:
- ✅ **`f1_cell`**: F1 score for this specific nuclei type
- ✅ **`prec_cell`**: Precision for this specific nuclei type
- ✅ **`rec_cell`**: Recall for this specific nuclei type

**`nuclei_metrics_pq`** - Segmentation Quality metrics:
- **PQ**: Panoptic Quality per nuclei type
- **DQ**: Detection Quality per nuclei type  
- **SQ**: Segmentation Quality per nuclei type

---

## Step-by-Step Example

### 1. Check Your Dataset Structure

Your dataset should have:
```
your_dataset/
├── dataset_config.yaml  # Contains nuclei_types and optionally tissue_types
├── train/
│   ├── images/
│   └── labels/  # .npy files with inst_map and type_map
├── test/
│   ├── images/
│   └── labels/
└── ...
```

### 2. Check Your Training Output

After training, you should have:
```
logs_local/
└── YourModel_YYYY_MM_DD_HHMMSS/
    ├── checkpoints/
    │   └── model_best.pth
    ├── config.yaml
    └── ...
```

### 3. Run Evaluation

**For PanNuke-style (with tissue types):**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir ./logs_local/YourModel_YYYY_MM_DD_HHMMSS \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

**For CoNSeP-style (no tissue types):**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_consep.py \
  --run_dir ./logs_local/YourModel_YYYY_MM_DD_HHMMSS \
  --checkpoint_name model_best.pth \
  --gpu 0
```

### 4. Review Results

Results are saved in your run directory:
```
logs_local/YourModel_YYYY_MM_DD_HHMMSS/
├── inference_results.json  # ← All metrics including per-class F1
└── inference.log           # ← Console output with formatted tables
```

**Check per-class F1 scores in JSON:**
```bash
cat logs_local/YourModel_YYYY_MM_DD_HHMMSS/inference_results.json
```

Look for the `nuclei_metrics_d` section for per-class F1, precision, and recall!

---

## Key Differences from Detection Dataset

| Aspect | Detection Dataset (CSV) | Segmentation Dataset (NumPy) |
|--------|------------------------|------------------------------|
| Annotation Format | CSV coordinates | NumPy instance masks |
| Evaluation Script | `inference_cellvit_experiment_detection.py` | `inference_cellvit_experiment_pannuke.py` |
| Per-Class F1 | `classifier.per_class[class]["f1"]` | `nuclei_metrics_d[class]["f1_cell"]` |
| Additional Metrics | AUROC, AP | PQ, DQ, SQ |
| Segmentation Quality | Approximate (generated) | Precise (from masks) |

---

## Summary

### Quick Answer

**File to use:** 
- `inference_cellvit_experiment_pannuke.py` (if tissue types)
- `inference_cellvit_experiment_consep.py` (if no tissue types)

**Per-class F1 location:**
- Console: "Nuclei Detection Metrics" table
- JSON: `nuclei_metrics_d[nuclei_type]["f1_cell"]`

**Both average and per-class F1 are automatically provided!**

---

## Troubleshooting

### "I don't know if I have tissue types"

Check your `dataset_config.yaml`:
```yaml
# Has tissue types (use pannuke.py)
tissue_types:
  Breast: 0
  Colon: 1
  ...
nuclei_types:
  Neoplastic: 1
  ...

# No tissue types (use consep.py)
nuclei_types:
  Neoplastic: 1
  Inflammatory: 2
  ...
```

### "Which magnification should I use?"

Check your training config or image resolution:
- 40x magnification: Most common for histopathology
- 20x magnification: Lower resolution images

When in doubt, use `--magnification 40` (default).

### "Can I see examples?"

Check the example segmentation datasets:
- `./test_database/training_database/Example-Segmentation/`
- `./test_database/training_database/Example-Segmentation-Non-Squared/`

---

## Additional Resources

- [Terminology Guide](TERMINOLOGY_GUIDE.md) - Understanding "detection" vs "segmentation"
- [Evaluation Guide](EVALUATION_GUIDE.md) - Comprehensive evaluation documentation
- [Understanding PanNuke Script](UNDERSTANDING_PANNUKE_SCRIPT.md) - Details on PanNuke evaluation

---

## Need Help?

If your output doesn't show per-class metrics, check:
1. ✅ Using the correct script (pannuke.py or consep.py for segmentation)
2. ✅ Your dataset has proper `dataset_config.yaml`
3. ✅ Your training used `SegmentationDataset` (not `DetectionDataset`)
4. ✅ Check the JSON file for `nuclei_metrics_d` section

The per-class F1 scores are **already built-in** - no modifications needed!
