# Quick Start: Evaluating Custom Classifiers

This quick start guide shows you how to evaluate a custom classifier trained with CellViT++.

## Prerequisites

✅ You have trained a classifier using `train_cell_classifier_head.py`  
✅ Your training completed successfully and created a log directory  
✅ You have the CellViT base model checkpoint  
✅ You have a test dataset in the same format as training

## Step 1: Locate Your Files

After training, you should have:

```
logs_local/
└── CellViT-Classifier_YYYY_MM_DD_HHMMSS/
    ├── checkpoints/
    │   └── model_best.pth          ← Your trained classifier
    ├── config.yaml                  ← Training configuration
    └── ... (other training files)
```

You also need:
- **Dataset path**: Parent directory of your dataset (e.g., `./test_database/training_database/Example-Detection`)
- **CellViT model**: Base segmentation model (e.g., `./checkpoints/CellViT-256-x40-AMP.pth`)
- **Input shape**: Image dimensions from training (check `config.yaml` or default is `256 256`)

## Step 2: Run Evaluation (Easy Mode)

Use the simplified evaluation script with built-in validation:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
  --logdir ./logs_local/CellViT-Classifier_YYYY_MM_DD_HHMMSS \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
```

The script will:
1. ✓ Validate all paths and files exist
2. ✓ Load your trained classifier
3. ✓ Run inference on test data
4. ✓ Calculate comprehensive metrics
5. ✓ Save results with clear messages

## Step 3: Review Results

Results are saved in `./logs_local/CellViT-Classifier_YYYY_MM_DD_HHMMSS/inference_results/`:

### Key Files

1. **`inference_results.json`** - All metrics in JSON format
   ```json
   {
     "global_classifier": {
       "auroc": 0.89,
       "f1": 0.82,
       "precision": 0.84,
       "recall": 0.81
     },
     "per_class": {
       "Tumor Cell": {"f1": 0.85, "precision": 0.87, "recall": 0.83},
       "sTIL": {"f1": 0.78, "precision": 0.80, "recall": 0.76}
     }
   }
   ```

2. **`confusion_matrix.png`** - Visual representation of classification performance

3. **`confusion_matrix_normalized.png`** - Normalized confusion matrix (percentages)

4. **`inference.log`** - Detailed execution log

### Understanding the Metrics

**AUROC (Area Under ROC Curve)**
- Range: 0 to 1 (higher is better)
- > 0.9: Excellent performance
- 0.7-0.9: Good performance
- < 0.7: May need improvement

**F1 Score**
- Balanced measure of precision and recall
- Good indicator of overall classification quality

**Confusion Matrix**
- Shows which classes are confused with each other
- Diagonal = correct predictions
- Off-diagonal = misclassifications

## Common Issues & Solutions

### "Training log directory not found"
**Problem**: `--logdir` path is incorrect  
**Solution**: Use the full path to your training output directory

### "Dataset path not found"
**Problem**: `--dataset_path` doesn't point to correct location  
**Solution**: Provide the parent directory containing `train/`, `test/`, and `splits/` folders

### "CellViT model not found"
**Problem**: Missing base model checkpoint  
**Solution**: Download from [Google Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79)

### "Input shape mismatch"
**Problem**: Using wrong dimensions  
**Solution**: Check your training `config.yaml` for `data.input_shape`

## Advanced Usage

### Using Stain Normalization

If you used stain normalization during training:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
  --logdir ./logs_local/your_run \
  --dataset_path ./your_dataset \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256 \
  --normalize_stains
```

### Using Different GPU

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
  --logdir ./logs_local/your_run \
  --dataset_path ./your_dataset \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256 \
  --gpu 1
```

### Direct Script (More Options)

For advanced users who want more control:

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir ./logs_local/your_run \
  --dataset_path ./your_dataset \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
```

Both scripts produce the same results, but the custom classifier script has better validation and error messages.

## Next Steps

1. **Review Metrics**: Check if performance meets your requirements
2. **Analyze Confusion Matrix**: Identify which classes need improvement
3. **Iterate if Needed**: Based on results, you may want to:
   - Collect more training data for poorly performing classes
   - Adjust class balance in training
   - Try different hyperparameters
   - Retrain with different augmentations

## Complete Example Workflow

```bash
# Step 1: Train classifier (already done)
python3 ./cellvit/train_cell_classifier_head.py --config your_config.yaml

# Step 2: Evaluate on test set
python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
  --logdir ./logs_local/CellViT-Classifier_2024_01_15_120000 \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256

# Step 3: Review results
cat ./logs_local/CellViT-Classifier_2024_01_15_120000/inference_results/inference_results.json

# Step 4: View confusion matrix
# Open inference_results/confusion_matrix.png
```

## Getting More Help

- **Detailed Guide**: See [docs/EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Script Comparison**: See [docs/UNDERSTANDING_PANNUKE_SCRIPT.md](UNDERSTANDING_PANNUKE_SCRIPT.md)
- **Training Guide**: See main [README.md](../README.md) section "Re-training your own classifier"

## Summary

For most users:
1. Use `inference_cellvit_custom_classifier.py` for easy evaluation
2. Provide: logdir, dataset_path, cellvit_path, input_shape
3. Review the JSON results and confusion matrices
4. Iterate on training if needed

That's it! The evaluation script handles the rest automatically.
