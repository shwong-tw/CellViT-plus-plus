# Understanding the Two-Stage Architecture: cellvit_path vs model_best.pth

## Quick Answer

**Q: Why do I need both `cellvit_path` and `model_best.pth`?**

**A: They are two different models doing two different jobs:**

- **`cellvit_path`**: Base CellViT model that **detects and segments cells** from images
- **`model_best.pth`**: Your trained classifier that **classifies** the detected cells into your custom nuclei types

You need both because the evaluation pipeline is:
```
Image → CellViT (detect cells) → Classifier (classify cells) → Results
```

---

## Detailed Explanation

### The Two Models

#### 1. CellViT Model (cellvit_path)

**What it is:**
- A pretrained deep learning model for cell segmentation
- Trained on large datasets (PanNuke, CoNSeP, etc.)
- Universal cell detector that works on various tissue types

**What it does:**
- Takes a whole-slide image as input
- Detects individual cell boundaries (instance segmentation)
- Extracts feature embeddings for each detected cell (768-dimensional vectors)

**Where you get it:**
- Download from CellViT model repository
- Pretrained checkpoints like `CellViT-256-x40.pth`
- You do NOT train this model yourself

**Example path:**
```
/path/to/models/CellViT-256-x40-AMP.pth
```

#### 2. Classifier Head (model_best.pth)

**What it is:**
- A small neural network (2-3 linear layers)
- Trained by YOU using `train_cell_classifier_head.py`
- Specific to YOUR nuclei types/classes

**What it does:**
- Takes cell embeddings from CellViT as input
- Classifies each cell into your custom classes (e.g., Tumor, Stromal, Immune)
- Outputs class probabilities and predictions

**Where you get it:**
- Automatically created during training
- Saved to `{logdir}/checkpoints/model_best.pth`
- This is what you trained!

**Example path:**
```
/path/to/training/run/checkpoints/model_best.pth
```

---

## The Complete Pipeline

### Visual Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Whole Slide Image (e.g., 1000x1000 pixels)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  STAGE 1: CellViT Model            │
        │  (cellvit_path)                    │
        │                                    │
        │  - Instance segmentation           │
        │  - Boundary detection              │
        │  - Feature extraction              │
        └────────────┬───────────────────────┘
                     │
                     │ Output: List of cells with embeddings
                     │ [Cell1: 768-dim vector,
                     │  Cell2: 768-dim vector,
                     │  Cell3: 768-dim vector, ...]
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STAGE 2: Classifier Head          │
        │  (model_best.pth in logdir)        │
        │                                    │
        │  - Takes 768-dim embeddings        │
        │  - Predicts nuclei type            │
        │  - Outputs class probabilities     │
        └────────────┬───────────────────────┘
                     │
                     │ Output: Class predictions
                     │ [Cell1: Tumor (0.95),
                     │  Cell2: Stromal (0.87),
                     │  Cell3: Immune (0.92), ...]
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ OUTPUT: Classified Cell Predictions│
        └────────────────────────────────────┘
```

### Code Walkthrough

In the evaluation scripts, here's how both models are loaded and used:

```python
# From inference_cellvit_experiment_classifier.py

# Line 144-145: Two separate paths
self.model_path = self.logdir / "checkpoints" / checkpoint_name  # Classifier
self.cellvit_path = Path(cellvit_path)  # CellViT base model

# Line 152-154: Load CellViT model (for cell detection)
self.cellvit_model, self.cellvit_run_conf = self._load_cellvit_model(
    checkpoint_path=self.cellvit_path  # ← Base segmentation model
)

# Line 155: Load Classifier model (for classification)
self.model, self.run_conf = self._load_model(
    checkpoint_path=self.model_path  # ← Your trained classifier
)
```

**_load_cellvit_model()** (lines 221-245):
- Loads the CellViT architecture (ViT-based segmentation model)
- Restores pretrained weights for cell detection
- Returns the full segmentation model

**_load_model()** (lines 197-219):
- Loads a LinearClassifier (small MLP)
- Restores YOUR trained weights for classification
- Returns the classification head only

---

## Why Both Are Essential

### Cannot Use Only CellViT

❌ **Problem:** CellViT doesn't know your custom classes

CellViT is a generic cell detector. It can find cells but:
- It doesn't know if a cell is "Tumor" or "Stromal" (your specific classes)
- It only provides generic cell features
- It has no information about your training data

### Cannot Use Only Classifier

❌ **Problem:** Classifier cannot process raw images

The classifier is trained to work with cell embeddings, not images:
- Input: 768-dimensional feature vectors (from CellViT)
- Output: Class predictions
- It has NO image processing capability
- It cannot detect or segment cells

### Why You Need Both

✅ **CellViT provides:** Cell detection + generic features
✅ **Classifier provides:** Your custom class predictions
✅ **Together:** Complete pipeline from image to classified cells

---

## Training vs Inference

### During Training (`train_cell_classifier_head.py`)

**What you train:**
- Only the classifier head (model_best.pth)
- CellViT is frozen (weights don't change)

**Process:**
1. Load pretrained CellViT (cellvit_path)
2. Extract cell embeddings from training images
3. Train classifier to predict your classes from embeddings
4. Save trained classifier to logdir/checkpoints/

**Files created:**
- `{logdir}/checkpoints/model_best.pth` ← Your trained classifier
- `{logdir}/checkpoints/latest_checkpoint.pth`
- `{logdir}/config.yaml`

### During Inference (Evaluation Scripts)

**What you load:**
- Both CellViT and classifier

**Process:**
1. Load CellViT from cellvit_path → for cell detection
2. Load classifier from logdir → for classification
3. Run pipeline: detect cells → classify cells
4. Compute metrics and save results

**Why both paths are needed:**
- CellViT: Could be anywhere you downloaded it
- Classifier: Always in {logdir}/checkpoints/

---

## Common Misconceptions

### ❌ "model_best.pth contains everything I need"

**Reality:** model_best.pth is just 2-3 small layers:
- Input layer: 768 → hidden_dim
- Hidden layer: hidden_dim → hidden_dim
- Output layer: hidden_dim → num_classes

Total size: ~1-5 MB (vs CellViT: ~300-800 MB)

### ❌ "I should retrain CellViT for my dataset"

**Reality:** You use transfer learning:
- CellViT is already excellent at detecting cells
- You only train the classifier on top
- Faster training, better results with less data

### ❌ "cellvit_path should point to logdir"

**Reality:** Two separate models, two separate paths:
- CellViT: Wherever you downloaded the pretrained model
- Classifier: logdir/checkpoints/model_best.pth

---

## Analogy

Think of it like a two-person team:

**Person 1 (CellViT):**
- Expert at finding cells in images
- "I found 500 cells in this image. Here are their features."
- Pretrained, never changes

**Person 2 (Classifier):**
- Expert at YOUR specific classification task
- "Cell #1's features look like Tumor. Cell #2 looks like Stromal."
- Trained by you on your data

Both work together:
1. Person 1 finds and describes the cells
2. Person 2 labels them with your classes

---

## FAQ

### Q: Can I use a different CellViT model?

**A:** Yes! You can use any CellViT variant:
- CellViT-256
- CellViT-SAM
- CellViT-UNI
- CellViT-Virchow

Just point cellvit_path to the appropriate checkpoint.

### Q: What if I trained multiple classifiers?

**A:** Use the `--checkpoint_name` parameter:
```bash
--checkpoint_name checkpoint_epoch_50.pth
```

This loads a different classifier checkpoint from the same logdir.

### Q: Do I need to retrain if I get a newer CellViT?

**A:** Generally no, but you could:
- Same CellViT version: No retraining needed
- Different CellViT variant: Might want to retrain classifier
- Feature dimensions must match (usually 768)

### Q: Can I train my own CellViT?

**A:** Yes, but it's complex and requires:
- Large datasets (thousands of annotated images)
- Segmentation masks (not just classification labels)
- Significant compute resources
- Different training script (not train_cell_classifier_head.py)

For classification, using pretrained CellViT is recommended.

---

## File Structure Example

```
/home/user/
├── models/
│   └── CellViT-256-x40-AMP.pth          ← cellvit_path (pretrained)
│
└── experiments/
    └── my_classifier_training/
        ├── checkpoints/
        │   ├── model_best.pth            ← Your trained classifier
        │   └── latest_checkpoint.pth
        ├── config.yaml
        └── test_results/                 ← Created during evaluation
            └── inference_results.json
```

**Command:**
```bash
python3 inference_cellvit_experiment_segmentation.py \
  --cellvit_path /home/user/models/CellViT-256-x40-AMP.pth \
  --logdir /home/user/experiments/my_classifier_training \
  --checkpoint_name model_best.pth \
  --dataset_path /data/test_dataset \
  --gpu 0
```

---

## Summary

| Aspect | cellvit_path | model_best.pth |
|--------|--------------|----------------|
| **Purpose** | Cell detection & segmentation | Cell classification |
| **Trained by** | CellViT authors (pretrained) | You (custom training) |
| **Size** | Large (300-800 MB) | Small (1-5 MB) |
| **Input** | Raw images | Cell embeddings |
| **Output** | Cell boundaries + embeddings | Class predictions |
| **Location** | Wherever downloaded | logdir/checkpoints/ |
| **Reusable** | Yes, across many projects | Specific to your classes |
| **Training time** | Days/weeks (if training from scratch) | Minutes/hours |

**Both are essential for the complete evaluation pipeline!**

---

## See Also

- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Complete evaluation reference
- [COMPLETE_WORKFLOW_EXAMPLE.md](COMPLETE_WORKFLOW_EXAMPLE.md) - End-to-end example
- [TERMINOLOGY_GUIDE.md](TERMINOLOGY_GUIDE.md) - Understanding terminology
