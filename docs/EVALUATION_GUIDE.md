# CellViT++ Classifier Evaluation Guide

This guide explains how to evaluate a custom classifier trained using `train_cell_classifier_head.py`.

## ⚠️ Important: Understanding "Detection" vs "Segmentation"

**Are you confused by these terms?** Read this first!

- **"Detection Dataset"** = CSV annotations with (x, y, label) coordinates
- **"Segmentation Dataset"** = NumPy annotations with instance masks and type labels

**Both formats support nuclei type classification!** The terms refer to **annotation format**, not the task.

> 📖 **Confused?** See [TERMINOLOGY_GUIDE.md](TERMINOLOGY_GUIDE.md) for detailed clarification.

**Quick decision:**
- Have CSV files with coordinates? → Use DetectionDataset & `inference_cellvit_experiment_detection.py`
- Have NumPy files with masks? → Use SegmentationDataset & `inference_cellvit_experiment_pannuke.py` (or similar)

---

## Overview

After training a classifier on your custom dataset, you need to evaluate its performance on test data. CellViT++ provides different evaluation scripts depending on your **annotation format** and requirements.

## Table of Contents

1. [Understanding the Evaluation Scripts](#understanding-the-evaluation-scripts)
2. [Which Script Should I Use?](#which-script-should-i-use)
3. [Evaluation Script Details](#evaluation-script-details)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Understanding the Metrics](#understanding-the-metrics)

---

## Understanding the Evaluation Scripts

CellViT++ provides several evaluation scripts in `./cellvit/training/evaluate/`:

> **Note:** "Detection" and "Segmentation" refer to **annotation format**, not task type!
> Both support multi-class nuclei type classification.

### 1. `inference_cellvit_experiment_detection.py` (For CSV Annotations)

**Purpose**: Evaluate classifiers trained on **CSV-based annotations** (point coordinates with labels)

**Key Features**:
- Works with custom class labels defined in your training config
- Calculates classification metrics (F1, Precision, Recall, AUROC) - both global and per-class
- Calculates detection quality metrics (how well cells are detected)
- Generates confusion matrices
- Provides both global and per-class performance metrics

**When to Use**:
- ✅ Your annotations are CSV files with (x, y, label) format
- ✅ You trained with `DetectionDataset`
- ✅ You want nuclei type classification metrics
- ✅ You have custom cell types/classes (e.g., Tumor, Stromal, Immune, etc.)

**What it evaluates:**
- Nuclei type classification (multi-class)
- Cell detection quality
- Per-class classification performance

### 2. `inference_cellvit_experiment_pannuke.py` (For NumPy Masks with Tissue Types)

**Purpose**: Evaluate models trained on **NumPy mask annotations** (PanNuke-style structure)

**Key Features**:
- Designed for NumPy files with instance masks and type labels
- Calculates segmentation metrics (Dice, Jaccard, PQ scores)
- Supports tissue types and nuclei types
- Requires specific dataset structure with tissue_types and nuclei_types
- **✅ Provides per-class F1 scores in `nuclei_metrics_d`**

**When to Use**:
- ✅ Your annotations are NumPy (.npy) files with instance masks
- ✅ You trained with `SegmentationDataset` (PanNuke-style)
- ✅ Your dataset has tissue types (e.g., breast, colon, etc.)
- ✅ Your dataset has a `dataset_config.yaml` with tissue_types and nuclei_types

**What it evaluates:**
- Nuclei type classification (multi-class)
- Instance segmentation quality (PQ, DQ, SQ)
- Tissue-specific performance
- **Per-nuclei-type F1, precision, recall** (in `nuclei_metrics_d`)

> 📖 **Quick guide for segmentation datasets:** See [SEGMENTATION_EVALUATION_QUICKSTART.md](SEGMENTATION_EVALUATION_QUICKSTART.md)

### 3. Other Segmentation Dataset Scripts

For NumPy mask annotations without tissue types:

**`inference_cellvit_experiment_consep.py`**: For CoNSeP-style datasets (nuclei types only, no tissue types)
- **✅ Also provides per-class F1 scores**

Scripts like `inference_cellvit_experiment_lizard.py`, etc., are designed for specific benchmark datasets with their evaluation protocols.

> 📖 **For segmentation datasets:** See [SEGMENTATION_EVALUATION_QUICKSTART.md](SEGMENTATION_EVALUATION_QUICKSTART.md) for details on per-class metrics.

---

## Which Script Should I Use?

Use this decision tree based on **annotation format**:

```
What format are your annotations in?

├─ CSV files with (x, y, label) coordinates
│  └─ Use: inference_cellvit_experiment_detection.py ✓
│     (Even if you're doing nuclei type classification!)
│
└─ NumPy (.npy) files with instance masks
   └─ Does your dataset have tissue types?
      ├─ YES → Use: inference_cellvit_experiment_pannuke.py
      │        (Requires dataset_config.yaml with tissue_types)
      │
      └─ NO → Use: inference_cellvit_experiment_consep.py
               (Or similar benchmark-specific script)
```

### Common Confusion Resolved

**❓ "I want to classify nuclei types - which script?"**
- **Answer:** Depends on your annotation format!
  - CSV coordinates → `inference_cellvit_experiment_detection.py`
  - NumPy masks → `inference_cellvit_experiment_pannuke.py` or `consep.py`

**❓ "I have segmentation masks with nuclei types - is that 'detection'?"**
- **Answer:** No! "Detection" means CSV format. You have segmentation format.
  - Use `inference_cellvit_experiment_pannuke.py` (if you have tissue types)
  - Or `inference_cellvit_experiment_consep.py` (if no tissue types)

**❓ "Does 'detection' script only do binary detection?"**
- **Answer:** No! It does multi-class nuclei type classification.
  - "Detection" refers to the annotation format (CSV), not the task
  - It calculates per-class F1, precision, recall for all your classes

**For most custom classifiers:** Check your training config's `dataset:` field
- `dataset: DetectionDataset` → Use `inference_cellvit_experiment_detection.py`
- `dataset: SegmentationDataset` → Use `inference_cellvit_experiment_pannuke.py` (or similar)

---

## Evaluation Script Details

### Using `inference_cellvit_experiment_detection.py`

#### Requirements

1. **Trained Classifier**: Directory with your trained classifier (output from `train_cell_classifier_head.py`)
2. **CellViT Model**: The base CellViT model checkpoint used during training
3. **Dataset**: Your dataset in the same structure as during training
4. **Input Shape**: The image dimensions used during training

#### Command Line Arguments

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir <path_to_trained_classifier> \
  --dataset_path <path_to_dataset> \
  --cellvit_path <path_to_cellvit_model> \
  --input_shape <height> <width> \
  [--normalize_stains] \
  [--gpu <gpu_id>]
```

**Arguments**:
- `--logdir`: Path to the directory containing your trained classifier (contains `checkpoints/` and `config.yaml`)
- `--dataset_path`: Path to your dataset parent directory (not the fold path)
- `--cellvit_path`: Path to the CellViT base model (.pth file) used during training
- `--input_shape`: Two integers for height and width (e.g., `256 256`)
- `--normalize_stains`: (Optional) Enable stain normalization if used during training
- `--gpu`: (Optional) GPU ID to use (default: 0)

#### Example

```bash
# Example with the provided Example-Detection dataset
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir ./logs_local/your_training_run \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
```

#### What the Script Does

1. **Loads Your Trained Classifier**: Loads the model from `logdir/checkpoints/model_best.pth`
2. **Loads the CellViT Model**: Loads the base segmentation model
3. **Loads Test Dataset**: Uses the test split defined in your dataset
4. **Performs Inference**:
   - Segments cells using CellViT
   - Classifies detected cells using your trained classifier
5. **Calculates Metrics**:
   - **Classification Metrics**: AUROC, F1, Precision, Recall, Accuracy
   - **Detection Metrics**: How well the pipeline detects cells (F1, Precision, Recall)
   - **Per-Class Metrics**: Performance for each cell type
6. **Generates Outputs**:
   - `inference_results.json`: Complete metrics in JSON format
   - `confusion_matrix.png`: Visualization of classification performance
   - `confusion_matrix_normalized.png`: Normalized confusion matrix
   - Log file with detailed results

#### Output Metrics Explained

The script produces several types of metrics:

**Global Classification Metrics** (without considering detection quality):
- **AUROC**: Area Under ROC Curve - overall classification performance (macro-averaged)
- **Macro F1 Score**: Harmonic mean of precision and recall averaged across all classes
- **Macro Precision**: Proportion of correct positive predictions averaged across classes
- **Macro Recall**: Proportion of actual positives correctly identified averaged across classes
- **Accuracy**: Overall correct predictions

**Per-Class Classification Metrics** (pure classification performance for each class):
- **F1 Score per Class**: F1 score calculated for each individual class
- **Precision per Class**: Precision for each class
- **Recall per Class**: Recall for each class

**Pipeline Detection Metrics** (TIA evaluation - considers both detection and classification):
- **Detection F1/Precision/Recall**: How well cells are detected overall
- **Per-Class Pipeline Metrics**: Detection + classification performance for each cell type in the full pipeline

**Confusion Matrix**:
- Shows which classes are confused with each other
- Normalized version shows proportions

### Using `inference_cellvit_experiment_pannuke.py`

#### Requirements

1. **Trained Model**: Directory with trained CellViT model (for PanNuke structure)
2. **Dataset Config**: Your dataset must have `dataset_config.yaml` with:
   ```yaml
   tissue_types:
     Adrenal_gland: 0
     Bile-duct: 1
     # ... other tissue types
   nuclei_types:
     Background: 0
     Neoplastic: 1
     Inflammatory: 2
     # ... other nuclei types
   ```

#### Command Line Arguments

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir <path_to_trained_model> \
  [--checkpoint_name <checkpoint_file>] \
  [--gpu <gpu_id>] \
  [--magnification <20|40>] \
  [--plots]
```

**Arguments**:
- `--run_dir`: Path to the training run directory
- `--checkpoint_name`: Name of checkpoint file (default: `model_best.pth`)
- `--gpu`: GPU ID (default: 0)
- `--magnification`: Dataset magnification, 20 or 40 (default: 40)
- `--plots`: Generate visualization plots

#### What the Script Does

1. **Loads the Trained Model**: Full CellViT model with segmentation
2. **Loads Test Dataset**: Uses PanNuke-style dataset structure
3. **Performs Inference**: Segments and classifies cells
4. **Calculates Metrics**:
   - **Binary Metrics**: Dice, Jaccard for cell detection
   - **PQ Scores**: Panoptic Quality (segmentation + classification)
   - **Tissue-Specific Metrics**: Performance per tissue type
   - **Nuclei Type Metrics**: Performance per nuclei class
5. **Generates Outputs**:
   - `inference_results.json`: All metrics
   - Optional visualization plots

---

## Step-by-Step Guide

### For Detection Dataset (Most Common Case)

#### Step 1: Complete Training

Train your classifier using the workflow from README section "Re-training your own classifier":

```bash
# Run hyperparameter sweep
python3 ./cellvit/train_cell_classifier_head.py --config your_config.yaml --sweep

# Find best configuration
python3 ./scripts/find_best_hyperparameter.py /path/to/sweep --metric AUROC/Validation

# Train final model
python3 ./cellvit/train_cell_classifier_head.py --config best_config.yaml
```

#### Step 2: Locate Your Training Output

After training, you'll have a log directory (e.g., `./logs_local/CellViT-Classifier_2024_01_15_120000/`) containing:
- `checkpoints/model_best.pth` - Your trained classifier
- `config.yaml` - Training configuration
- Training logs and metrics

#### Step 3: Prepare for Evaluation

You need:
1. ✓ Path to your training log directory
2. ✓ Path to your dataset (parent directory)
3. ✓ Path to CellViT base model used during training
4. ✓ Input shape used during training (check `config.yaml` → `data.input_shape` or default `[256, 256]`)

#### Step 4: Run Evaluation

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir ./logs_local/CellViT-Classifier_2024_01_15_120000 \
  --dataset_path /path/to/your/dataset \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
```

#### Step 5: Review Results

Check the outputs in your `logdir/inference_results/`:
- `inference_results.json` - Numerical metrics
- `confusion_matrix.png` - Visual performance assessment
- `inference.log` - Detailed logs

Key metrics to review:
- **AUROC**: Should be > 0.8 for good performance
- **F1 Score**: Balance of precision and recall
- **Per-Class Metrics**: Identify which classes perform well/poorly
- **Confusion Matrix**: See which classes are confused

---

## Understanding the Metrics

### Classification Metrics (Global)

These metrics evaluate the classifier's performance assuming perfect cell detection:

- **AUROC (Area Under ROC Curve)**: 
  - Range: 0 to 1 (higher is better)
  - Measures overall discriminative ability
  - > 0.9: Excellent, 0.7-0.9: Good, < 0.7: Poor

- **F1 Score**:
  - Range: 0 to 1 (higher is better)
  - Harmonic mean of precision and recall
  - Best when you need balance between precision and recall

- **Precision**:
  - What proportion of positive predictions are correct?
  - Important when false positives are costly

- **Recall (Sensitivity)**:
  - What proportion of actual positives are found?
  - Important when false negatives are costly

### Detection Metrics (Pipeline)

These metrics evaluate the full pipeline (detection + classification):

- **Detection F1/Precision/Recall**:
  - Measures how well cells are detected overall
  - Uses distance-based matching between ground truth and predictions

- **Per-Class Detection Metrics**:
  - Evaluates detection and classification together for each class
  - Shows which cell types are harder to find/classify

### Segmentation Metrics (PanNuke Script)

- **Dice Coefficient**: Overlap between predicted and ground truth segments
- **Jaccard Index**: Intersection over union for segments
- **PQ (Panoptic Quality)**: Combines segmentation and detection quality
- **DQ (Detection Quality)**: How well instances are detected
- **SQ (Segmentation Quality)**: How well instances are segmented

---

## Troubleshooting

### Common Issues

**Error: "Cannot find checkpoint"**
- Ensure `--logdir` points to the correct training output directory
- Check that `model_best.pth` exists in `logdir/checkpoints/`

**Error: "Input shape mismatch"**
- Verify `--input_shape` matches what was used during training
- Check your `config.yaml` file for the correct shape

**Error: "Dataset path not found"**
- Ensure `--dataset_path` points to the parent dataset directory
- The directory should contain `test/images/` and `test/labels/`

**Low performance metrics**
- Check if test data distribution matches training data
- Verify class labels are consistent
- Consider retraining with more data or different hyperparameters
- Review confusion matrix to understand misclassifications

**GPU out of memory**
- Reduce batch size in the evaluation script (may need to modify script)
- Use smaller model or lower resolution
- Use a GPU with more memory

---

## Additional Resources

- Training Guide: See README section "Re-training your own classifier"
- Dataset Preparation: See `./logs/Datasets/` for examples
- Example Datasets: `./test_database/training_database/Example-Detection/`
- Script Source: `./cellvit/training/evaluate/`

---

## Summary

**For most users with custom classifiers:**

1. Train using `train_cell_classifier_head.py`
2. Evaluate using `inference_cellvit_experiment_detection.py`
3. Review metrics in `inference_results.json` and confusion matrices
4. Iterate on training if needed based on results

The detection evaluation script is designed to work seamlessly with custom classifiers and provides comprehensive metrics to assess your model's performance.
