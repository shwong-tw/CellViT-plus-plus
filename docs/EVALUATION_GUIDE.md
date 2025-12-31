# CellViT++ Classifier Evaluation Guide

This guide explains how to evaluate a custom classifier trained using `train_cell_classifier_head.py`.

## Overview

After training a classifier on your custom dataset, you need to evaluate its performance on test data. CellViT++ provides different evaluation scripts depending on your dataset type and requirements.

## Table of Contents

1. [Understanding the Evaluation Scripts](#understanding-the-evaluation-scripts)
2. [Which Script Should I Use?](#which-script-should-i-use)
3. [Evaluation Script Details](#evaluation-script-details)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Understanding the Metrics](#understanding-the-metrics)

---

## Understanding the Evaluation Scripts

CellViT++ provides several evaluation scripts in `./cellvit/training/evaluate/`:

### 1. `inference_cellvit_experiment_detection.py` (Recommended for Custom Classifiers)

**Purpose**: Evaluate classifiers trained on detection datasets (CSV annotations with x, y coordinates)

**Key Features**:
- Works with custom class labels defined in your training config
- Calculates classification metrics (F1, Precision, Recall, AUROC)
- Calculates detection quality metrics (how well cells are detected)
- Generates confusion matrices
- Provides both global and per-class performance metrics

**When to Use**:
- You trained with `DetectionDataset` (CSV annotations with x, y coordinates and labels)
- You want detailed classification performance metrics
- You have custom cell types/classes

### 2. `inference_cellvit_experiment_pannuke.py`

**Purpose**: Evaluate models specifically on PanNuke dataset structure

**Key Features**:
- Designed for PanNuke dataset with predefined tissue types and nuclei types
- Calculates segmentation metrics (Dice, Jaccard, PQ scores)
- Requires specific dataset structure with tissue types

**When to Use**:
- You trained on PanNuke dataset or a dataset with the same structure
- You need tissue-specific metrics
- Your dataset has a `dataset_config.yaml` with tissue_types and nuclei_types

### 3. Other Dataset-Specific Scripts

Scripts like `inference_cellvit_experiment_consep.py`, `inference_cellvit_experiment_lizard.py`, etc., are designed for specific benchmark datasets with their evaluation protocols.

---

## Which Script Should I Use?

Use this decision tree:

```
Did you train with DetectionDataset (CSV annotations)?
├─ YES → Use inference_cellvit_experiment_detection.py ✓
└─ NO
   └─ Did you train with SegmentationDataset (NumPy masks)?
      ├─ YES → Is your dataset PanNuke or has tissue types?
      │  ├─ YES → Use inference_cellvit_experiment_pannuke.py
      │  └─ NO → Use inference_cellvit_experiment_consep.py (or similar)
      └─ NO → Contact for support
```

**For most custom classifiers trained using the workflow in the README: Use `inference_cellvit_experiment_detection.py`**

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
- **AUROC**: Area Under ROC Curve - overall classification performance
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **Accuracy**: Overall correct predictions

**Pipeline Detection Metrics** (TIA evaluation - considers both detection and classification):
- **Detection F1/Precision/Recall**: How well cells are detected
- **Per-Class F1/Precision/Recall**: Performance for each cell type in the full pipeline

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
