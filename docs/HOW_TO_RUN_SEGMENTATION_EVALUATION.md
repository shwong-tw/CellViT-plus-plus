# How to Run Segmentation Evaluation

This guide shows you **exactly** how to run the evaluation script for your nuclei-only segmentation dataset.

## Quick Start

### 1. The Command

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir /path/to/your/training/run \
  --cellvit_path /path/to/CellViT-model.pth \
  --dataset_path /path/to/your/dataset \
  --checkpoint_name model_best.pth \
  --split test \
  --gpu 0
```

### 2. Required Arguments

You **MUST** provide these arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--logdir` | Path to your training output directory (contains `checkpoints/` folder) | `./logs/my_classifier_20240101` |
| `--cellvit_path` | Path to the pretrained CellViT base model | `./models/CellViT-256-x40.pth` |
| `--dataset_path` | Path to your dataset root folder | `./data/my_nuclei_dataset` |

### 3. Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_name` | `model_best.pth` | Which checkpoint to evaluate |
| `--split` | `test` | Which data split to evaluate (e.g., test, val, Test) |
| `--label_map_file` | `label_map.yaml` | Name of the label config file |
| `--gt_format` | `npy` | Ground truth format: `npy` or `mat` |
| `--normalize_stains` | `False` | Whether to normalize stains (use flag to enable) |
| `--gpu` | `0` | GPU device number to use |

---

## Complete Example

### Your Setup

Let's say you have:
- **Training output**: `/home/user/experiments/my_classifier`
- **CellViT model**: `/home/user/models/CellViT-256-x40.pth`
- **Dataset**: `/home/user/data/nuclei_dataset`
- **Dataset structure**:
  ```
  /home/user/data/nuclei_dataset/
  ├── test/
  │   ├── images/
  │   │   ├── image001.png
  │   │   └── image002.png
  │   └── labels/
  │       ├── image001.npy
  │       └── image002.npy
  └── label_map.yaml
  ```

### Command to Run

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir /home/user/experiments/my_classifier \
  --cellvit_path /home/user/models/CellViT-256-x40.pth \
  --dataset_path /home/user/data/nuclei_dataset \
  --checkpoint_name model_best.pth \
  --split test \
  --gpu 0
```

---

## Common Errors and Solutions

### Error: "FileNotFoundError: [Errno 2] No such file or directory"

**Cause**: One of your paths doesn't exist.

**Solution**: Check each path:
```bash
# Check training directory exists
ls -la /path/to/your/training/run

# Check CellViT model exists
ls -la /path/to/CellViT-model.pth

# Check dataset exists
ls -la /path/to/your/dataset
```

### Error: "error: the following arguments are required: --logdir, --dataset_path, --cellvit_path"

**Cause**: You're missing required arguments.

**Solution**: Make sure you provide ALL three required arguments:
- `--logdir`
- `--dataset_path`
- `--cellvit_path`

### Error: "ModuleNotFoundError: No module named 'cv2'"

**Cause**: Missing OpenCV dependency.

**Solution**: Install required packages:
```bash
pip install opencv-python
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Error: "RuntimeError: No CUDA GPUs are available"

**Cause**: No GPU available or wrong GPU number.

**Solution**: 
- Check available GPUs: `nvidia-smi`
- Use CPU instead: `--gpu -1`
- Or use correct GPU number: `--gpu 0`

---

## Understanding the Output

### Console Output

You'll see:
```
2026-02-09 10:30:12,703 [INFO] - Step 1: Extracting cells with CellViT
2026-02-09 10:30:15,432 [INFO] - Step 2: Classifying cells with classifier
2026-02-09 10:30:18,123 [INFO] - Global Classifier Scores (without detection quality):
2026-02-09 10:30:18,124 [INFO] - F1: 0.857 - Prec: 0.872 - Rec: 0.842 - Acc: 0.865 - Auroc: 0.892 - AP: 0.885

******************** Nuclei Detection Metrics ********************
Nuclei Type          Precision      Recall          F1
----------------------------------------------------------------
Tumor                   0.872        0.845       0.858
Stromal                 0.823        0.798       0.810
Immune                  0.791        0.774       0.782
----------------------------------------------------------------
Average                 0.829        0.806       0.817
```

### JSON Output

Results saved to: `{logdir}/test_results/inference_results.json`

```json
{
  "cellvit_scores": {
    "F1": 0.85,
    "Prec": 0.87,
    "Rec": 0.83
  },
  "classifier": {
    "global": {
      "F1": 0.857,
      "Prec": 0.872,
      "Rec": 0.842,
      "Acc": 0.865,
      "Auroc": 0.892,
      "AP": 0.885
    }
  },
  "pipeline": {
    "segmentation_scores": {
      "dice": 0.845,
      "aji": 0.678,
      "aji_plus": 0.712
    },
    "detection_scores": {
      "cell_types": {
        "Tumor": {
          "f1": 0.858,
          "precision": 0.872,
          "recall": 0.845
        },
        "Stromal": {
          "f1": 0.810,
          "precision": 0.823,
          "recall": 0.798
        },
        "Immune": {
          "f1": 0.782,
          "precision": 0.791,
          "recall": 0.774
        }
      }
    },
    "pq_scores": {
      "pq": 0.745,
      "cell_types+": {
        "Tumor": {
          "pq": 0.768,
          "dq": 0.845,
          "sq": 0.908
        },
        "Stromal": {
          "pq": 0.732,
          "dq": 0.812,
          "sq": 0.901
        },
        "Immune": {
          "pq": 0.698,
          "dq": 0.778,
          "sq": 0.897
        }
      }
    }
  }
}
```

### Cell Predictions

Individual cell predictions saved to: `{logdir}/test_results/cell_predictions/`
- One JSON file per image
- Contains bounding boxes, centroids, types, probabilities

---

## Step-by-Step Workflow

### Before Running

1. ✅ Complete training using `train_cell_classifier_head.py`
2. ✅ Find your training output directory (contains `checkpoints/model_best.pth`)
3. ✅ Locate your CellViT base model
4. ✅ Prepare your test dataset with proper structure

### Run Evaluation

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir <your_training_dir> \
  --cellvit_path <cellvit_model> \
  --dataset_path <dataset_root> \
  --split test \
  --gpu 0
```

### After Running

1. ✅ Check console for per-class F1 scores
2. ✅ Review JSON results in `{logdir}/test_results/inference_results.json`
3. ✅ Analyze per-class metrics to identify which cell types need improvement
4. ✅ Check confusion matrix: `{logdir}/test_results/confusion_matrix.png`

---

## Getting Help

If you still have issues:

1. **Check the documentation**:
   - [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Comprehensive guide
   - [SOLUTION_SEGMENTATION_NO_TISSUE_TYPES.md](SOLUTION_SEGMENTATION_NO_TISSUE_TYPES.md) - Dataset setup
   - [UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md](UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md) - Why both models needed

2. **Verify your setup**:
   ```bash
   # Show help and all arguments
   python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py --help
   ```

3. **Check dataset structure**:
   ```bash
   # Your dataset should have:
   cd /path/to/dataset
   ls -la test/images/      # Image files
   ls -la test/labels/      # Ground truth .npy or .mat files
   ls -la label_map.yaml    # Or dataset_config.yaml
   ```

4. **Verify training output**:
   ```bash
   # Your training directory should have:
   cd /path/to/training/run
   ls -la checkpoints/model_best.pth   # Trained classifier
   ls -la config.yaml                   # Training config
   ```

---

## Quick Reference

**Minimal command (all defaults)**:
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir ./logs/my_run \
  --cellvit_path ./models/CellViT.pth \
  --dataset_path ./data/my_dataset
```

**Full command (all options)**:
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py \
  --logdir ./logs/my_run \
  --cellvit_path ./models/CellViT-256-x40.pth \
  --dataset_path ./data/my_dataset \
  --checkpoint_name model_best.pth \
  --split test \
  --label_map_file label_map.yaml \
  --gt_format npy \
  --normalize_stains \
  --gpu 0
```

**See all options**:
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py --help
```
