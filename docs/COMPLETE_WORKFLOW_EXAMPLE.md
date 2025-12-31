# Complete Workflow Example: Training and Evaluating a Custom Classifier

This document provides a complete, end-to-end example of training and evaluating a custom cell classifier with CellViT++.

## Overview

We'll walk through:
1. Dataset preparation
2. Training configuration
3. Hyperparameter search
4. Finding best model
5. **Evaluation** (focus of this guide)
6. Interpreting results

## Prerequisites

- ✅ CellViT++ installed and environment activated
- ✅ Example dataset downloaded
- ✅ CellViT base model checkpoint downloaded

## Complete Example

### Step 1: Prepare Your Dataset

Your dataset should have this structure:

```
Example-Detection/
├── label_map.yaml           # Optional: maps class IDs to names
├── splits/                   # Train/val splits
│   ├── fold_0/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── fold_1/
│   │   ├── train.csv
│   │   └── val.csv
│   └── ...
├── train/                    # Training data
│   ├── images/              # PNG/JPG images
│   │   ├── train_001.png
│   │   └── ...
│   └── labels/              # CSV annotations
│       ├── train_001.csv    # Format: x, y, label
│       └── ...
├── test/                     # Test data (same structure)
│   ├── images/
│   └── labels/
└── train_configs/           # Training configurations
    └── ViT256/
        ├── fold_0.yaml
        └── fold_0_sweep.yaml
```

**Label format in CSV** (train_001.csv):
```csv
x,y,label
150,200,0
300,450,1
...
```

**Label map** (label_map.yaml):
```yaml
0: "Tumor Cell"
1: "Stromal Cell"
2: "Immune Cell"
3: "Other"
```

### Step 2: Create Training Configuration

Create `train_configs/ViT256/fold_0.yaml`:

```yaml
logging:
  mode: offline
  project: cellvit++
  notes: Custom classifier training
  log_comment: my_custom_classifier
  wandb_dir: ./logs_local
  log_dir: ./logs_local
  level: Debug

random_seed: 19
gpu: 0

data:
  dataset: DetectionDataset
  dataset_path: ./test_database/training_database/Example-Detection
  normalize_stains_train: false
  normalize_stains_val: false
  num_classes: 4  # Your number of classes
  train_filelist: ./test_database/training_database/Example-Detection/splits/fold_0/train.csv
  val_filelist: ./test_database/training_database/Example-Detection/splits/fold_0/val.csv
  label_map:
    0: Tumor Cell
    1: Stromal Cell
    2: Immune Cell
    3: Other

cellvit_path: ./checkpoints/CellViT-256-x40-AMP.pth

model:
  hidden_dim: 256

training:
  cache_cell_dataset: true
  batch_size: 64
  epochs: 50
  drop_rate: 0.1
  optimizer: AdamW
  optimizer_hyperparameter:
    betas: [0.85, 0.9]
    lr: 0.0002609902875925979
    weight_decay: 1.8529312539791538e-05
  early_stopping_patience: 20
  mixed_precision: true
  eval_every: 1
  scheduler:
    scheduler_type: exponential
```

### Step 3: Run Hyperparameter Search (Optional but Recommended)

Create sweep config `fold_0_sweep.yaml` for hyperparameter search:

```yaml
# Same as above, but with sweep parameters
sweep:
  method: bayes
  metric:
    name: AUROC/Validation
    goal: maximize
  parameters:
    drop_rate:
      values: [0.0, 0.1, 0.2]
    optimizer_hyperparameter:
      lr:
        min: 0.0001
        max: 0.001
      weight_decay:
        min: 0.00001
        max: 0.0001
```

Run the sweep:

```bash
# Create logs directory
mkdir -p logs_local

# Start hyperparameter sweep
python3 ./cellvit/train_cell_classifier_head.py \
  --config ./test_database/training_database/Example-Detection/train_configs/ViT256/fold_0_sweep.yaml \
  --sweep

# This will prompt you to login to Weights & Biases
# Choose option 1 or 2 (not 3)
```

### Step 4: Find Best Configuration

After sweep completes (may take hours depending on search space):

```bash
python3 ./scripts/find_best_hyperparameter.py \
  ./logs_local/sweeps/your_sweep_id \
  --metric AUROC/Validation
```

This outputs the best configuration file path.

### Step 5: Train Final Model (Optional)

If you did hyperparameter search, train with best config:

```bash
python3 ./cellvit/train_cell_classifier_head.py \
  --config /path/to/best_config.yaml
```

Otherwise, your sweep already trained models, use the best one.

### Step 6: Evaluate Your Model ⭐

**This is where the new evaluation documentation helps!**

#### Option A: Simplified Script (Recommended)

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
  --logdir ./logs_local/CellViT-Classifier_2024_01_15_120000 \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
```

**What happens:**
1. ✅ Validates all paths exist
2. ✅ Checks for required files
3. ✅ Loads your trained classifier
4. ✅ Runs inference on test set
5. ✅ Calculates comprehensive metrics
6. ✅ Saves results with clear messages

**Output:**
```
╔════════════════════════════════════════════════════════════════════════════╗
║          CellViT++ Custom Classifier Evaluation                            ║
╚════════════════════════════════════════════════════════════════════════════╝

================================================================================
VALIDATING INPUTS
================================================================================
✓ Log directory found: ./logs_local/CellViT-Classifier_2024_01_15_120000
✓ Configuration file found: ...
✓ Model checkpoint found: ...
✓ Dataset directory found: ...
✓ Test data found: ...
✓ CellViT model found: ...
✓ Input shape: 256 x 256
================================================================================
VALIDATION PASSED - Ready to run evaluation
================================================================================

Starting inference on test set...
[Progress bar and inference details]

================================================================================
EVALUATION COMPLETED SUCCESSFULLY!
================================================================================
Results saved to: ./logs_local/.../inference_results
Generated files:
  - inference_results.json
  - confusion_matrix.png
  - confusion_matrix_normalized.png
  - inference.log
================================================================================
```

#### Option B: Direct Script (More Control)

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir ./logs_local/CellViT-Classifier_2024_01_15_120000 \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
```

Both produce identical results.

### Step 7: Review Results

#### A. Check JSON Metrics

```bash
cat ./logs_local/CellViT-Classifier_2024_01_15_120000/inference_results/inference_results.json
```

Example output:
```json
{
  "global_classifier": {
    "auroc": 0.8923,
    "f1": 0.8245,
    "precision": 0.8456,
    "recall": 0.8045,
    "accuracy": 0.8534
  },
  "per_class": {
    "Tumor Cell": {
      "f1": 0.8567,
      "precision": 0.8723,
      "recall": 0.8415,
      "auroc": 0.9123
    },
    "Stromal Cell": {
      "f1": 0.7892,
      "precision": 0.8124,
      "recall": 0.7675,
      "auroc": 0.8745
    },
    "Immune Cell": {
      "f1": 0.8334,
      "precision": 0.8501,
      "recall": 0.8175,
      "auroc": 0.8956
    },
    "Other": {
      "f1": 0.7123,
      "precision": 0.7456,
      "recall": 0.6812,
      "auroc": 0.8234
    }
  },
  "pipeline": {
    "detection_scores_tia": {
      "f1_detection": 0.7845,
      "precision_detection": 0.8123,
      "recall_detection": 0.7589,
      "cell_types": {
        "Tumor Cell": {
          "f1": 0.8234,
          "precision": 0.8456,
          "recall": 0.8023
        }
        // ... other classes
      }
    }
  }
}
```

#### B. View Confusion Matrix

Open the confusion matrix images:
- `confusion_matrix.png` - Raw counts
- `confusion_matrix_normalized.png` - Percentages (easier to interpret)

**What to look for:**
- Strong diagonal (correct predictions)
- Off-diagonal patterns (which classes are confused)

#### C. Interpret Results

**AUROC > 0.9**: Excellent! Your classifier is performing very well.

**AUROC 0.7-0.9**: Good performance, may benefit from:
- More training data
- Better class balance
- Hyperparameter tuning

**AUROC < 0.7**: Needs improvement:
- Check data quality
- Review class definitions
- Consider more training data
- Try different model architecture

**Confusion Matrix Insights:**
- High off-diagonal between "Stromal" and "Other"? → These classes may be too similar
- One class consistently misclassified? → Need more examples or better features

### Step 8: Iterate if Needed

Based on results:

**If performance is good**: ✅ Done! Use your classifier for inference.

**If specific classes perform poorly**:
1. Collect more training examples for those classes
2. Review label quality
3. Check if class definitions are clear

**If overall performance is low**:
1. Try different hyperparameters
2. Increase training data
3. Use data augmentation
4. Consider different base model

## Complete Command Reference

```bash
# 1. Create logs directory
mkdir -p logs_local

# 2. Run hyperparameter sweep
python3 ./cellvit/train_cell_classifier_head.py \
  --config ./test_database/training_database/Example-Detection/train_configs/ViT256/fold_0_sweep.yaml \
  --sweep

# 3. Find best hyperparameters
python3 ./scripts/find_best_hyperparameter.py \
  ./logs_local/sweeps/your_sweep_id \
  --metric AUROC/Validation

# 4. Train final model (if needed)
python3 ./cellvit/train_cell_classifier_head.py \
  --config /path/to/best_config.yaml

# 5. Evaluate on test set (SIMPLIFIED - RECOMMENDED)
python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
  --logdir ./logs_local/CellViT-Classifier_YYYY_MM_DD_HHMMSS \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256

# 6. Review results
cat ./logs_local/CellViT-Classifier_YYYY_MM_DD_HHMMSS/inference_results/inference_results.json
```

## Tips and Best Practices

### Training
- Use hyperparameter sweep for best results
- Monitor validation metrics during training
- Use early stopping to prevent overfitting
- Cache dataset for faster training

### Evaluation
- Always use test set (not validation) for final evaluation
- Check both global and per-class metrics
- Review confusion matrix for insights
- Compare results across different folds if using cross-validation

### Debugging
- If validation fails, check all file paths
- Ensure input_shape matches training
- Verify dataset structure is correct
- Check that CellViT model path is correct

## Additional Resources

- **Evaluation Guide**: [docs/EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Quick Start**: [docs/EVALUATION_QUICKSTART.md](EVALUATION_QUICKSTART.md)
- **Understanding Scripts**: [docs/UNDERSTANDING_PANNUKE_SCRIPT.md](UNDERSTANDING_PANNUKE_SCRIPT.md)
- **Main README**: [README.md](../README.md)

## Summary

This complete workflow shows:
1. ✅ Dataset preparation
2. ✅ Training configuration
3. ✅ Hyperparameter search
4. ✅ Model training
5. ✅ **Evaluation with new simplified script**
6. ✅ Results interpretation
7. ✅ Iteration for improvement

The new evaluation documentation and script make Step 6 much easier with clear validation, helpful errors, and comprehensive metrics!
