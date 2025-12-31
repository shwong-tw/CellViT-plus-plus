# Evaluation Documentation Index

This directory contains comprehensive documentation for evaluating custom CellViT++ classifiers.

## Quick Links

### 🚀 New to Evaluation?
Start here → [**Evaluation Quick Start**](EVALUATION_QUICKSTART.md)

### 📖 Need Detailed Information?
Comprehensive guide → [**Evaluation Guide**](EVALUATION_GUIDE.md)

### ❓ Confused About PanNuke Script?
Understanding guide → [**Understanding PanNuke Script**](UNDERSTANDING_PANNUKE_SCRIPT.md)

### 📋 Developer Reference
Summary of changes → [**Evaluation Summary**](EVALUATION_SUMMARY.md)

---

## Document Overview

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

### Scenario 1: I just trained a custom classifier
→ Use [EVALUATION_QUICKSTART.md](EVALUATION_QUICKSTART.md)

### Scenario 2: I'm not sure which evaluation script to use
→ See decision tree in [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#which-script-should-i-use)

### Scenario 3: I need to understand the PanNuke script
→ Read [UNDERSTANDING_PANNUKE_SCRIPT.md](UNDERSTANDING_PANNUKE_SCRIPT.md)

### Scenario 4: I'm getting errors during evaluation
→ Check troubleshooting in [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#troubleshooting)

### Scenario 5: I want to understand all the metrics
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
