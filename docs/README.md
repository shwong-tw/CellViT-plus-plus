# Evaluation Documentation Index

This directory contains comprehensive documentation for evaluating custom CellViT++ classifiers.

## ⭐ MOST COMMON QUESTION

**"My training dataset is with segmentation annotation. I've done the retraining. Which file do I use to run evaluation on test dataset and get per-class F1 scores?"**

### Step 1: Do you have tissue type labels?

- **NO (only nuclei classes)** → [**Use the Generic Nuclei Segmentation Script**](SOLUTION_SEGMENTATION_NO_TISSUE_TYPES.md) ← **Most common! RECOMMENDED**
  - Script: `inference_cellvit_experiment_segmentation.py`
  - No dummy data needed, works out of the box!
  
- **YES (both tissue and nuclei types)** → [**Direct Answer for Segmentation WITH Tissue Types**](DIRECT_ANSWER_SEGMENTATION_EVALUATION.md)
  - Script: `inference_cellvit_experiment_pannuke.py`

**TL;DR:**
- Only nuclei: Use new generic script `inference_cellvit_experiment_segmentation.py` (clean, no workarounds needed)
- With tissue: Use PanNuke script `inference_cellvit_experiment_pannuke.py`

---

## ⚠️ Confused About Terminology?

**Start here if unclear about "detection" vs "segmentation":**
→ [**Terminology Guide**](TERMINOLOGY_GUIDE.md) - Explains what these terms actually mean

**TL;DR:**
- "Detection" = CSV annotation format (still does nuclei type classification!)
- "Segmentation" = NumPy mask format (also does nuclei type classification!)
- Both support multi-class classification - choose based on annotation format!

---

## Quick Links

### 🚀 **HOW TO RUN THE SCRIPT** ⭐
**Need to run evaluation NOW?** → [**How to Run Segmentation Evaluation**](HOW_TO_RUN_SEGMENTATION_EVALUATION.md)
- Complete command examples
- All required arguments explained
- Common errors and solutions
- Step-by-step workflow

### ⭐ Direct Answer for Segmentation Dataset
**Most asked question** → [**Direct Answer: Segmentation Evaluation**](DIRECT_ANSWER_SEGMENTATION_EVALUATION.md)

### 🚀 New to Evaluation?
Start here → [**Evaluation Quick Start**](EVALUATION_QUICKSTART.md)

### 📊 Have Segmentation Dataset? Need Per-Class F1?
Quick answer → [**Segmentation Evaluation Quick Start**](SEGMENTATION_EVALUATION_QUICKSTART.md)

### 📖 Need Detailed Information?
Comprehensive guide → [**Evaluation Guide**](EVALUATION_GUIDE.md)

### ❓ Confused About PanNuke Script?
Understanding guide → [**Understanding PanNuke Script**](UNDERSTANDING_PANNUKE_SCRIPT.md)

### 🔤 Confused About Terms?
Terminology clarification → [**Terminology Guide**](TERMINOLOGY_GUIDE.md)

### 🔍 Understanding "Without Taking Detection Into Account"
Detailed explanation → [**Detection vs Classification Metrics**](UNDERSTANDING_DETECTION_VS_CLASSIFICATION_METRICS.md)

### 🏗️ Understanding the Two-Stage Architecture
**Why both cellvit_path and model_best.pth?** → [**Two-Stage Architecture**](UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md)

### 📋 Developer Reference
Summary of changes → [**Evaluation Summary**](EVALUATION_SUMMARY.md)

---

## Document Overview

### SOLUTION_SEGMENTATION_NO_TISSUE_TYPES.md (⭐ NEW! - Most Requested)
**Best for**: Segmentation datasets with ONLY nuclei classes (no tissue types)

**Contains**:
- Two practical solutions (dummy tissue type OR custom script)
- Step-by-step walkthrough with examples
- Quick diagnostic to determine which approach
- Full example with dataset_config.yaml
- Troubleshooting common errors

**Reading time**: 10 minutes

### HOW_TO_RUN_SEGMENTATION_EVALUATION.md (⭐ NEW! ESSENTIAL)
**Best for**: Anyone who needs to actually RUN the evaluation script

**Contains**:
- **EXACT command to run** with all required arguments
- Complete example with real paths
- Every argument explained with examples
- Common errors and how to fix them
- Understanding the output (console + JSON)
- Step-by-step workflow from training to evaluation
- Quick reference section

**Reading time**: 15-20 minutes (or 2 minutes if you just need the command)

### DIRECT_ANSWER_SEGMENTATION_EVALUATION.md (⭐)
**Best for**: Segmentation datasets WITH tissue types - Direct answer to common question

**Contains**:
- Step-by-step command to run
- Expected output with per-class F1 scores
- Troubleshooting common issues
- Quick verification checklist
- Now includes check for tissue types

**Reading time**: 5 minutes

### SEGMENTATION_EVALUATION_QUICKSTART.md
**Best for**: Users with segmentation datasets who need per-class F1 scores

**Contains**:
- Direct answer: which script to use for segmentation datasets
- Example of per-class F1 output (console and JSON)
- Step-by-step evaluation workflow
- Difference between PanNuke-style and CoNSeP-style
- Troubleshooting tips

**Reading time**: 5-10 minutes

### UNDERSTANDING_DETECTION_VS_CLASSIFICATION_METRICS.md (NEW!)
**Best for**: Understanding what "without taking detection into account" means

**Contains**:
- Explanation of two-stage evaluation (detection + classification)
- How "without detection" metrics isolate classifier performance
- Visual comparison of the two evaluation modes
- Code implementation details
- Use cases for each metric type
- Debugging guide using both metrics

**Reading time**: 15-20 minutes

### TERMINOLOGY_GUIDE.md (NEW!)
**Best for**: Anyone confused about "detection" vs "segmentation" terminology

**Contains**:
- Clear explanation of what these terms mean
- Why both support nuclei type classification
- Decision trees based on annotation format
- Common confusion scenarios resolved
- Quick reference tables

**Reading time**: 10-15 minutes

### EVALUATION_QUICKSTART.md
**Best for**: Users who want to evaluate their custom classifier quickly

**Contains**:
- Prerequisites checklist
- 3-step evaluation process
- Common issues and quick solutions
- Complete example workflow

**Reading time**: 5-10 minutes

### EVALUATION_GUIDE.md
**Best for**: Comprehensive understanding of all evaluation options

**Contains**:
- Detailed overview of all evaluation scripts
- Decision tree for choosing the right script
- Complete CLI argument documentation
- Metrics explanations (AUROC, F1, PQ, DQ, SQ, etc.)
- Troubleshooting guide
- Step-by-step workflows for different scenarios

**Reading time**: 20-30 minutes

### UNDERSTANDING_PANNUKE_SCRIPT.md
**Best for**: Understanding or modifying the PanNuke evaluation script

**Contains**:
- Explanation of PanNuke script purpose
- Key components and their requirements
- What needs modification for custom classes
- Comparison with detection script
- Recommendations on when to use it

**Reading time**: 10-15 minutes

### UNDERSTANDING_DETECTION_VS_CLASSIFICATION_METRICS.md
**Best for**: Understanding the phrase "without taking detection into account"

**Contains**:
- Explanation of two-stage evaluation (detection + classification)
- How classification metrics ignore detection quality
- Code walkthrough of cell pairing process
- Why you get two different F1 scores
- Use cases for each metric type

**Reading time**: 10-15 minutes

### UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md (⭐ NEW!)
**Best for**: Understanding why both cellvit_path and model_best.pth are needed

**Contains**:
- Explanation of two-model architecture (CellViT + Classifier)
- Visual diagram of the pipeline
- Code walkthrough showing where each model is loaded
- Why you cannot use only one model
- Training vs inference model usage
- Common misconceptions and FAQ

**Reading time**: 10-15 minutes

### EVALUATION_SUMMARY.md
**Best for**: Developers and maintainers

**Contains**:
- Overview of all changes made
- Problem statement and solution
- File descriptions
- User journey mapping
- Testing status

**Reading time**: 15-20 minutes

---

## Common Scenarios

### Scenario 0: ⭐ I have segmentation dataset and need per-class F1 scores (MOST COMMON)
→ **[DIRECT_ANSWER_SEGMENTATION_EVALUATION.md](DIRECT_ANSWER_SEGMENTATION_EVALUATION.md)** - Direct step-by-step answer!

### Scenario 0.5: I'm confused about "detection" vs "segmentation"
→ Read [TERMINOLOGY_GUIDE.md](TERMINOLOGY_GUIDE.md) first!

### Scenario 1: I just trained a custom classifier
→ Use [EVALUATION_QUICKSTART.md](EVALUATION_QUICKSTART.md)

### Scenario 2: I'm not sure which evaluation script to use
→ Check your annotation format:
- CSV files → See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#which-script-should-i-use) section on DetectionDataset
- NumPy masks → See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#which-script-should-i-use) section on SegmentationDataset

### Scenario 3: I have segmentation masks with nuclei types
→ You need a SegmentationDataset evaluation script!
- Read [TERMINOLOGY_GUIDE.md](TERMINOLOGY_GUIDE.md#scenario-2-i-have-segmentation-masks-with-nuclei-types)
- Then use `inference_cellvit_experiment_pannuke.py` or `consep.py`

### Scenario 4: I need to understand the PanNuke script
→ Read [UNDERSTANDING_PANNUKE_SCRIPT.md](UNDERSTANDING_PANNUKE_SCRIPT.md)

### Scenario 5: What does "without taking detection into account" mean?
→ Read [UNDERSTANDING_DETECTION_VS_CLASSIFICATION_METRICS.md](UNDERSTANDING_DETECTION_VS_CLASSIFICATION_METRICS.md)

### Scenario 6: Why do I need both cellvit_path and model_best.pth?
→ Read [UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md](UNDERSTANDING_TWO_STAGE_ARCHITECTURE.md)

### Scenario 7: I'm getting errors during evaluation
→ Check troubleshooting in [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#troubleshooting)

### Scenario 8: I want to understand all the metrics
→ See metrics section in [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#understanding-the-metrics)

---

## Evaluation Scripts Reference

Located in `./cellvit/training/evaluate/`:

### For Custom Classifiers (Detection Data)

1. **inference_cellvit_custom_classifier.py** (Recommended)
   - Simplified interface with validation
   - Best for beginners
   - Clear error messages

2. **inference_cellvit_experiment_detection.py**
   - Full-featured version
   - Same functionality as above
   - More advanced options

### For Benchmark Datasets

3. **inference_cellvit_experiment_pannuke.py**
   - PanNuke dataset specific
   - Requires tissue types and nuclei types
   - Calculates PQ/DQ/SQ metrics

4. **Other dataset-specific scripts**
   - CoNSeP, Lizard, OCELOT, etc.
   - Each optimized for specific benchmark

---

## Getting Started

**3-Step Quick Start:**

1. **Read** → [EVALUATION_QUICKSTART.md](EVALUATION_QUICKSTART.md)
2. **Run** → Use the simplified script
   ```bash
   python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \
     --logdir ./logs_local/your_run \
     --dataset_path ./your_dataset \
     --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
     --input_shape 256 256
   ```
3. **Review** → Check results in `logdir/inference_results/`

---

## Need Help?

- Check [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) troubleshooting section
- Review the script's `--help` output
- Ensure all paths and files are correct
- Verify dataset structure matches training

---

## Related Documentation

- **Training Guide**: See main [README.md](../README.md) section "Re-training your own classifier"
- **Dataset Preparation**: See `./logs/Datasets/` for examples
- **Example Datasets**: `./test_database/training_database/`

---

## Version Information

These documents were created for CellViT++ to help users evaluate custom classifiers trained with `train_cell_classifier_head.py`.

Last updated: 2024

---

## Feedback

If you find issues with the documentation or have suggestions for improvement, please open an issue on the GitHub repository.
