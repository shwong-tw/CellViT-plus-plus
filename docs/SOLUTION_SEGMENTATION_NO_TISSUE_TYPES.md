# SOLUTION: Segmentation Dataset with Only Nuclei Classes (No Tissue Types)

## Your Situation

- ✅ Segmentation annotation dataset (NumPy masks)
- ✅ Nuclei class labels (e.g., Tumor, Non-Tumor, Inflammatory, etc.)
- ❌ **NO tissue type labels**
- ✅ Completed retraining
- ❓ Need per-class F1 scores

## The Problem

The `inference_cellvit_experiment_pannuke.py` script expects **both** tissue types AND nuclei types because it's designed for the PanNuke dataset structure.

## The Solution

You have **two options** depending on your dataset structure:

---

## Option 1: Use PanNuke Script with Workaround (Recommended if it works)

The PanNuke script CAN work without tissue types if you provide a dummy tissue type configuration.

### Step 1: Check Your Dataset Configuration

Look at your training run's `config.yaml`:

```bash
cat /path/to/your/training/run/config.yaml
```

Check if it has a `dataset_config` section that looks like this:

```yaml
dataset_config:
  nuclei_types:
    0: Background
    1: Tumor
    2: Non-Tumor
    3: Inflammatory
    # etc.
```

### Step 2: Add Dummy Tissue Types (if needed)

If your `dataset_config.yaml` doesn't have tissue types, you can add a single dummy tissue type:

**Create or edit `dataset_config.yaml` in your dataset folder:**

```yaml
nuclei_types:
  0: Background
  1: Tumor
  2: Non-Tumor
  3: Inflammatory
  # Add all your nuclei classes here

tissue_types:
  0: "default"  # Dummy tissue type - all images belong to this
```

This tells the script that all your images belong to one tissue type called "default", so it will just evaluate nuclei types.

### Step 3: Run Evaluation

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir /path/to/your/training/run \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

### What to Expect

The script will:
- Process all images as tissue type "default"
- Calculate **per-class F1 scores** for each nuclei type
- Ignore tissue-specific metrics (since there's only one tissue type)

**Output:**
```
******************** Nuclei Detection Metrics ********************
Nuclei Type          Precision      Recall          F1
----------------------------------------------------------------
Tumor                   0.872        0.845       0.858    ← Your nuclei classes
Non-Tumor               0.823        0.798       0.810
Inflammatory            0.791        0.774       0.782
----------------------------------------------------------------
Average                 0.829        0.806       0.817
```

---

## Option 2: Create a Custom Evaluation Script

If Option 1 doesn't work or your dataset structure is very different, you'll need to create a custom evaluation script.

### Approach A: Adapt the PanNuke Script

1. **Copy the PanNuke script:**
```bash
cp ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
   ./cellvit/training/evaluate/inference_my_custom_segmentation.py
```

2. **Modify the script to remove tissue type dependencies:**
   - Remove tissue type loading
   - Remove tissue-specific metrics calculation
   - Focus only on nuclei type metrics

3. **Run your custom script:**
```bash
python3 ./cellvit/training/evaluate/inference_my_custom_segmentation.py \
  --run_dir /path/to/your/training/run \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

### Approach B: Check if You Actually Have Detection Data

If your annotations are actually in CSV format (x, y, label) rather than NumPy masks:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir /path/to/your/training/run \
  --dataset_path /path/to/your/dataset \
  --cellvit_path /path/to/cellvit/model.pth \
  --checkpoint_name model_best.pth \
  --input_shape 256 256 \
  --gpu 0
```

This script provides per-class F1 scores for CSV-based annotations.

---

## Quick Diagnostic: Which Approach to Use?

Answer these questions:

### 1. What format are your annotations?

**A) NumPy files (.npy) with instance masks**
→ You have segmentation data - continue to question 2

**B) CSV files with (x, y, label) coordinates**
→ Use `inference_cellvit_experiment_detection.py` (see Approach B above)

### 2. Do you have a dataset_config.yaml file?

**Yes, and it has nuclei_types:**
```yaml
nuclei_types:
  0: Background
  1: Class1
  2: Class2
```
→ Try **Option 1** (add dummy tissue types)

**No, or missing nuclei_types:**
→ Try **Option 2** (custom script)

### 3. What does your training config show?

Check your training run's `config.yaml`:

```bash
cat /path/to/your/training/run/config.yaml | grep -A 20 "dataset_config"
```

**If you see `tissue_types` in the config:**
→ Your dataset already has tissue types! Use PanNuke script directly

**If you only see `nuclei_types`:**
→ Try **Option 1** (add dummy tissue type)

**If you don't see dataset_config:**
→ You likely have detection data, try **Approach B**

---

## Example: Full Walkthrough for Option 1

### Scenario
- You have segmentation annotations
- 3 nuclei classes: Tumor, Stromal, Immune
- No tissue types

### Step 1: Create dataset_config.yaml

In your dataset folder, create `dataset_config.yaml`:

```yaml
nuclei_types:
  0: Background
  1: Tumor
  2: Stromal
  3: Immune

tissue_types:
  0: "all"  # Dummy - all images are this tissue type
```

### Step 2: Update Your Training Config (if needed)

If your training run's `config.yaml` doesn't reference this, you may need to ensure the dataset configuration is loaded. Check:

```bash
cat ./logs/your_training_run/config.yaml
```

### Step 3: Run Evaluation

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir ./logs/your_training_run \
  --checkpoint_name model_best.pth \
  --gpu 0 \
  --magnification 40
```

### Step 4: Check Results

Results will be in `./logs/your_training_run/test_results/inference_results.json`:

```json
{
  "nuclei_metrics_d": {
    "Tumor": {
      "f1_cell": 0.858,
      "prec_cell": 0.872,
      "rec_cell": 0.845
    },
    "Stromal": {
      "f1_cell": 0.810,
      "prec_cell": 0.823,
      "rec_cell": 0.798
    },
    "Immune": {
      "f1_cell": 0.782,
      "prec_cell": 0.791,
      "rec_cell": 0.774
    }
  },
  "dataset": {
    "f1_detection": 0.817  // Average F1
  }
}
```

---

## Troubleshooting

### Error: "KeyError: 'tissue_types'"

**Cause:** The script is looking for tissue types in your dataset config but can't find them.

**Solution:** Add a dummy tissue type to your `dataset_config.yaml` (see Option 1, Step 2)

### Error: "Dataset not found" or "Dataset path incorrect"

**Cause:** The script can't find your test dataset.

**Solution:** The PanNuke script loads the dataset automatically from the run configuration. Make sure your training run's `config.yaml` has the correct dataset path.

### Error: Different error message

**Solution:** You may need to create a custom evaluation script (Option 2). Please provide:
- The exact error message
- Your dataset folder structure
- Your training config.yaml content

---

## Summary

**If you have segmentation annotations WITHOUT tissue types:**

1. **Try this first:** Add a dummy tissue type to your dataset_config.yaml
2. **Run:** `inference_cellvit_experiment_pannuke.py`
3. **Get:** Per-class F1 scores for each nuclei type

**If that doesn't work:**
- Create a custom evaluation script based on PanNuke
- Or verify if you actually have detection data (CSV format)

**Files to check:**
- Your training run's `config.yaml`
- Your dataset's `dataset_config.yaml` or `label_map.yaml`
- Your annotation file format (.npy vs .csv)

---

## Need More Help?

If you're still stuck, please provide:

1. **Your annotation format:**
   ```bash
   ls -la /path/to/your/dataset/train/
   ls -la /path/to/your/dataset/train/labels/ | head -5
   ```

2. **Your training config:**
   ```bash
   cat /path/to/your/training/run/config.yaml | grep -A 50 "data:"
   ```

3. **Any error messages** you're getting

This will help diagnose the exact issue and provide a precise solution.
