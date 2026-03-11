# DIRECT ANSWER: Evaluating Segmentation Dataset with Per-Class F1 Scores

## Your Question

> "My training dataset is with segmentation annotation. I've done the retraining. Now which is the file I use to run evaluation on test dataset and provides per-class F1 score apart from original avg F1 score?"

## ⚠️ IMPORTANT: Do You Have Tissue Types?

**STOP!** Before proceeding, answer this question:

**Does your dataset have tissue type labels?**
- **Tissue types** = different tissue categories (e.g., breast, colon, prostate)
- **Nuclei types** = different cell types (e.g., tumor, immune, stromal)

### 🔴 If you have ONLY nuclei classes (NO tissue types):
→ **[Click here for your solution](SOLUTION_SEGMENTATION_NO_TISSUE_TYPES.md)** ←

This is common! Many datasets only have cell type labels without tissue classification.

### 🟢 If you have BOTH tissue types AND nuclei types:
→ Continue reading below

---

## Direct Answer (For Datasets WITH Tissue Types)

**USE THIS SCRIPT:** `inference_cellvit_experiment_pannuke.py`

**WHY:** This script already provides per-class F1 scores for segmentation datasets with NumPy mask annotations.

---

## Step-by-Step Instructions

### Step 1: Locate Your Training Run Directory

After retraining, you should have a directory structure like:
```
logs/
└── your_training_run_name/
    ├── checkpoints/
    │   ├── model_best.pth
    │   ├── latest_checkpoint.pth
    │   └── ...
    ├── config.yaml
    └── ...
```

### Step 2: Run Evaluation

**Command:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir /path/to/your/training/run \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

**Replace:**
- `/path/to/your/training/run` → Your actual training run directory (e.g., `./logs/my_segmentation_training`)
- `--gpu 0` → Your GPU number (or omit if using CPU)
- `--magnification 40` → Your dataset magnification (20 or 40, check your training config)

**Full Example:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir ./logs/segmentation_training_2024_01_15 \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

### Step 3: Check Output

The script will create a test results directory:
```
logs/your_training_run_name/test_results/
├── inference_results.json    # ← Contains per-class F1 scores
├── plots/
└── ...
```

---

## Output: Per-Class F1 Scores

### Console Output

You will see:
```
******************** Nuclei Detection Metrics ********************
Nuclei Type          Precision      Recall          F1
----------------------------------------------------------------
Neoplastic              0.872        0.845       0.858    ← Per-class F1
Inflammatory            0.823        0.798       0.810    ← Per-class F1
Connective              0.791        0.774       0.782    ← Per-class F1
Dead                    0.745        0.712       0.728    ← Per-class F1
Epithelial              0.834        0.821       0.827    ← Per-class F1
----------------------------------------------------------------
Average                 0.813        0.790       0.801    ← Average F1
```

### JSON Output

In `inference_results.json`:
```json
{
  "nuclei_metrics_d": {
    "Neoplastic": {
      "f1_cell": 0.858,          // ← Per-class F1 for Neoplastic
      "prec_cell": 0.872,
      "rec_cell": 0.845,
      "binary_dice": 0.863,
      "pq": 0.712,
      "sq": 0.854,
      "dq": 0.834
    },
    "Inflammatory": {
      "f1_cell": 0.810,          // ← Per-class F1 for Inflammatory
      "prec_cell": 0.823,
      "rec_cell": 0.798,
      ...
    },
    "Connective": {
      "f1_cell": 0.782,          // ← Per-class F1 for Connective
      ...
    },
    ...
  },
  "dataset": {
    "f1_detection": 0.801,       // ← Average F1 across all classes
    "Binary-Cell-Dice-Mean": 0.851,
    "bPQ": 0.632,
    "mPQ": 0.598
  }
}
```

**Key Points:**
- ✅ **Per-class F1**: `nuclei_metrics_d[class_name]["f1_cell"]`
- ✅ **Average F1**: `dataset["f1_detection"]`
- ✅ **Both are automatically calculated and displayed**

---

## Common Issues and Solutions

### Issue 1: "I don't have tissue types"

**Solution:** The PanNuke script can work without tissue types. It will evaluate on all nuclei types in your dataset.

**What to check:**
- Does your `dataset_config.yaml` have a `nuclei_types` section?
- If yes, the script will work
- If no, you may need to add it

**Example `dataset_config.yaml`:**
```yaml
nuclei_types:
  0: Background
  1: Neoplastic
  2: Inflammatory
  3: Connective
  4: Dead
  5: Epithelial
```

### Issue 2: "Script fails with dataset not found"

**Check your run directory structure:**
```bash
ls -la /path/to/your/training/run/
```

Should contain:
- `checkpoints/` directory
- `config.yaml` file
- Dataset configuration

**Fix:** Use the correct path to your training run directory.

### Issue 3: "I get different F1 scores than expected"

**Possible reasons:**
1. Using wrong checkpoint (use `model_best.pth` for best validation performance)
2. Wrong magnification setting
3. Different preprocessing settings

**Solution:**
```bash
# Check your training config
cat /path/to/your/training/run/config.yaml

# Use the same magnification as training
# Use model_best.pth for best results
```

### Issue 4: "I want to use a different checkpoint"

**Solution:** Use the `--checkpoint_name` parameter:
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir ./logs/my_run \
  --checkpoint_name checkpoint_100.pth \  # ← Different checkpoint
  --gpu 0 \
  --magnification 40
```

---

## Alternative: If PanNuke Script Doesn't Work

If the PanNuke script doesn't work for your dataset structure, you have these options:

### Option A: Check if you trained with DetectionDataset (CSV annotations)

If you used CSV annotations with (x, y, label) format:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir /path/to/your/training/run \
  --dataset_path /path/to/your/dataset \
  --cellvit_path /path/to/cellvit/model.pth \
  --checkpoint_name model_best.pth \
  --input_shape 256 256 \
  --gpu 0
```

This also provides per-class F1 scores (see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)).

### Option B: Adapt the PanNuke script to your dataset

If your dataset structure is different, you may need to:
1. Copy `inference_cellvit_experiment_pannuke.py`
2. Modify the dataset loading section
3. Adjust the metrics calculation if needed

---

## Quick Verification Checklist

Before running evaluation:

- [ ] I have a training run directory with `checkpoints/` folder
- [ ] I have `model_best.pth` or another checkpoint file
- [ ] I have a `config.yaml` in my run directory
- [ ] I know my dataset magnification (usually 20 or 40)
- [ ] My dataset has NumPy mask annotations (not CSV)
- [ ] I'm using the correct GPU number (or 0 for default)

After running evaluation:

- [ ] Console shows "Nuclei Detection Metrics" table with per-class F1
- [ ] `test_results/inference_results.json` file is created
- [ ] JSON contains `nuclei_metrics_d` with per-class metrics
- [ ] JSON contains `dataset` with average metrics

---

## Summary

**Question:** Which file to use for segmentation dataset evaluation with per-class F1?

**Answer:** `inference_cellvit_experiment_pannuke.py`

**Command:**
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir /path/to/your/training/run \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

**Output:**
- ✅ Per-class F1 scores for each nuclei type
- ✅ Average F1 score across all classes
- ✅ Additional segmentation metrics (PQ, Dice, etc.)

**Location of per-class F1 in JSON:**
```json
{
  "nuclei_metrics_d": {
    "YourClassName": {
      "f1_cell": 0.XXX  // ← This is the per-class F1
    }
  }
}
```

---

## Still Having Issues?

1. **Check the logs:** Look for error messages in the console output
2. **Verify paths:** Make sure all paths are absolute and correct
3. **Check dataset format:** Confirm you have NumPy segmentation masks
4. **Review training config:** `cat /path/to/run/config.yaml`
5. **Consult other docs:**
   - [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Comprehensive guide
   - [SEGMENTATION_EVALUATION_QUICKSTART.md](SEGMENTATION_EVALUATION_QUICKSTART.md) - Quick reference
   - [TERMINOLOGY_GUIDE.md](TERMINOLOGY_GUIDE.md) - Understanding terms

If you continue to have issues, please provide:
- Your exact command
- Error messages (if any)
- Training dataset format
- Directory structure of your training run
