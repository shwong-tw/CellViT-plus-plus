# Troubleshooting Common Evaluation Errors

This guide helps you resolve common errors when running CellViT++ evaluation scripts.

---

## Table of Contents

1. [Error: "2026-XX-XX XX:XX:XX,XXX [INFO] -" (Incomplete Log)](#error-incomplete-log)
2. [Error: "ModuleNotFoundError: No module named 'cv2'"](#error-missing-cv2)
3. [Error: "FileNotFoundError: CellViT model not found"](#error-cellvit-not-found)
4. [Error: "FileNotFoundError: Classifier checkpoint not found"](#error-checkpoint-not-found)
5. [Error: "Split folder not found"](#error-split-not-found)
6. [Error: "CUDA out of memory"](#error-cuda-memory)
7. [General Tips](#general-tips)

---

## Error: Incomplete Log

### Symptoms
```bash
$ python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py
2026-03-09 00:51:43,138 [INFO] -
(message cuts off)
```

### Cause
Missing Python dependencies. The script tries to import packages that aren't installed, causing Python to exit before the error message can be displayed.

### Solution
The script now checks dependencies before importing. You should see:

```
======================================================================
🛑 Missing Required Dependencies
======================================================================

The following Python packages are required but not installed:

   ❌ cv2 (install with: pip install opencv-python)
   ❌ torch (install with: pip install torch torchvision)
   ...

======================================================================
To install all missing packages, run:

   pip install opencv-python torch torchvision tqdm scikit-learn torchmetrics pycm

======================================================================
```

**Fix:** Run the suggested pip install command.

---

## Error: Missing cv2

### Symptoms
```
ModuleNotFoundError: No module named 'cv2'
```

### Cause
OpenCV (cv2) is not installed in your Python environment.

### Solution
```bash
pip install opencv-python
```

For additional features:
```bash
pip install opencv-contrib-python
```

---

## Error: CellViT Not Found

### Symptoms
```
======================================================================
🛑 Path Validation Failed
======================================================================

❌ CellViT model not found: ./checkpoints/CellViT-256-x40-AMP.pth
   This is the pretrained CellViT segmentation model.
   Download from: https://github.com/TIO-IKIM/CellViT/releases
```

### Cause
The pretrained CellViT base model file doesn't exist at the specified path.

### Solution

1. **Download the model:**
   - Visit: https://github.com/TIO-IKIM/CellViT/releases
   - Download the appropriate CellViT model (e.g., `CellViT-256-x40-AMP.pth`)

2. **Place it in your project:**
   ```bash
   mkdir -p ./checkpoints/HIPT-25
   # Move downloaded file to ./checkpoints/HIPT-25/CellViT-256-x40-AMP.pth
   ```

3. **Verify the path in your command:**
   ```bash
   --cellvit_path ./checkpoints/HIPT-25/CellViT-256-x40-AMP.pth
   ```

---

## Error: Checkpoint Not Found

### Symptoms
```
❌ Classifier checkpoint not found: ./logs/my_run/checkpoints/model_best.pth
   Expected location: ./logs/my_run/checkpoints/model_best.pth
   Available checkpoints:
      - checkpoint_50.pth
      - checkpoint_100.pth
```

### Cause
The specified checkpoint file doesn't exist. Maybe:
- Training didn't complete
- File was saved with a different name
- Wrong `--checkpoint_name` specified

### Solution

1. **Check available checkpoints:**
   ```bash
   ls ./logs/my_run/checkpoints/
   ```

2. **Use the correct checkpoint name:**
   ```bash
   --checkpoint_name checkpoint_100.pth
   ```

3. **If no checkpoints exist, you need to train the model first:**
   - See: `./cellvit/train_cell_classifier_head.py`

---

## Error: Split Not Found

### Symptoms
```
❌ Split folder not found: ./data/my_dataset/test
   Expected: ./data/my_dataset/test/
   Available splits:
      - train
      - val
```

### Cause
The dataset doesn't have the requested split folder.

### Solution

1. **Check your dataset structure:**
   ```bash
   ls ./data/my_dataset/
   ```

2. **Use an existing split:**
   ```bash
   --split val  # Instead of --split test
   ```

3. **Or create the test split:**
   ```bash
   mkdir -p ./data/my_dataset/test/images
   mkdir -p ./data/my_dataset/test/labels
   # Copy your test data into these folders
   ```

---

## Error: CUDA Memory

### Symptoms
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
```

### Cause
GPU doesn't have enough memory for the current batch size.

### Solutions

1. **Use CPU instead:**
   ```bash
   --gpu -1  # Force CPU usage
   ```

2. **Reduce batch size** (if the script supports it):
   ```bash
   --batch_size 1  # Or smaller value
   ```

3. **Use a smaller model:**
   - Try CellViT-SAM-H instead of CellViT-256

4. **Clear GPU memory:**
   ```python
   # In Python console
   import torch
   torch.cuda.empty_cache()
   ```

---

## General Tips

### 1. Always Check Dependencies First

Before running any evaluation script, ensure all dependencies are installed:

```bash
pip install opencv-python torch torchvision tqdm scikit-learn torchmetrics pycm pyyaml matplotlib
```

### 2. Verify Your Dataset Structure

Make sure your dataset follows the expected structure:

```
dataset_path/
├── test/                    # Your split folder
│   ├── images/
│   │   └── *.png, *.jpg
│   └── labels/
│       └── *.npy or *.mat
└── label_map.yaml           # Or dataset_config.yaml
```

### 3. Use `--help` to See All Options

```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_segmentation.py --help
```

### 4. Check File Paths

Use absolute paths to avoid confusion:

```bash
--logdir /absolute/path/to/logs/my_run
--cellvit_path /absolute/path/to/checkpoints/CellViT-256-x40-AMP.pth
```

### 5. Read the Documentation

- **Quick Start:** `docs/HOW_TO_RUN_SEGMENTATION_EVALUATION.md`
- **Architecture:** `docs/UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md`
- **Complete Guide:** `docs/EVALUATION_GUIDE.md`

### 6. Enable Verbose Logging

Most scripts support verbose output that can help identify issues.

---

## Still Having Issues?

If you're still encountering problems:

1. **Check the error message carefully** - our scripts now provide detailed, actionable error messages

2. **Verify your environment:**
   ```bash
   python3 --version  # Should be Python 3.8+
   pip list | grep torch  # Check PyTorch is installed
   pip list | grep cv2    # Check OpenCV is installed
   ```

3. **Review the documentation:**
   - Start with `docs/HOW_TO_RUN_SEGMENTATION_EVALUATION.md`
   - Check `docs/EVALUATION_GUIDE.md` for comprehensive information

4. **Look at example commands** in the documentation - they show working configurations

---

## Quick Reference: Installation

### Minimal Installation
```bash
pip install opencv-python torch torchvision tqdm scikit-learn torchmetrics pycm pyyaml matplotlib
```

### With GPU Support
Visit https://pytorch.org/get-started/locally/ to get the correct PyTorch installation command for your CUDA version.

Example for CUDA 11.8:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python tqdm scikit-learn torchmetrics pycm pyyaml matplotlib
```

---

## Summary

Most evaluation errors fall into three categories:

1. **Missing dependencies** → Install required packages
2. **Wrong paths** → Verify file locations
3. **Dataset structure** → Follow expected layout

The enhanced error messages now guide you to the exact fix needed. If you see a formatted error message with suggestions, follow the instructions provided!
