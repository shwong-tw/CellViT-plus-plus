# Understanding inference_cellvit_experiment_monuseg.py

This document explains the MoNuSeg evaluation script to help you understand when and how to use it.

## ⚠️ Important Notice

**This script is designed specifically for the MoNuSeg dataset benchmark.**

MoNuSeg = **Mo**ulti-organ **Nu**clei **Seg**mentation

**Key Characteristic:** MoNuSeg is a **BINARY segmentation** task - it only detects nuclei without classifying them into types.

If you trained a classifier for **multi-class classification**, this script is NOT suitable. Use:
- `inference_cellvit_experiment_segmentation.py` for nuclei-only segmentation with types
- `inference_cellvit_experiment_detection.py` for detection datasets (CSV annotations)
- `inference_cellvit_experiment_pannuke.py` for tissue + nuclei type classification

Only use this script if:
- You're evaluating on the MoNuSeg benchmark specifically
- You're doing binary segmentation (nuclei vs background, no types)

## Script Purpose

This script evaluates CellViT models on the MoNuSeg dataset for **binary nuclei segmentation**:
1. **No Classification**: Only detects nuclei boundaries, doesn't classify types
2. **Multi-organ**: Images from different tissue types (breast, kidney, liver, etc.)
3. **Patch-based Processing**: Supports large images with optional patching and overlap
4. **Standalone Architecture**: Does NOT inherit from CellViTClassifierInferenceExperiment

## Key Differences from Other Scripts

| Feature | MoNuSeg Script | CoNSeP Script | Segmentation Script |
|---------|----------------|---------------|---------------------|
| **Task** | Binary segmentation | Multi-class segmentation | Multi-class segmentation |
| **Nuclei Types** | No (just detect nuclei) | Yes (4 types) | Yes (configurable) |
| **Inheritance** | Standalone class | Extends base class | Extends base class |
| **Patching Support** | Yes (built-in) | No | No |
| **Overlap Handling** | Yes (advanced) | No | No |
| **Model Loading** | Direct CellViT | Via base class | Via base class |
| **Use Case** | Binary segmentation benchmark | Classification benchmark | Custom classification |

## Architecture: Standalone vs Inherited

**MoNuSeg Script is DIFFERENT:**

```python
# MoNuSeg - Standalone class
class MoNuSegInference:
    def __init__(self, model_path, dataset_path, outdir, gpu, ...):
        self.__load_model()  # Direct model loading
        self.inference_dataset = MoNuSegDataset(...)  # Direct dataset creation
```

**Other Scripts - Inherited:**

```python
# CoNSeP, Segmentation - Extend base class
class CellViTInfExpCoNSep(CellViTClassifierInferenceExperiment):
    def _load_dataset(self, ...):  # Override method
        return CoNSePDataset(...)
```

**Why Different:**
- MoNuSeg is **binary segmentation** - no classifier head needed
- Other scripts are for **classification** - need classifier head
- MoNuSeg needs **advanced patching** for large images
- Simpler, more direct implementation for its specific task

## Key Components

### 1. Initialization (Lines 63-111)

```python
class MoNuSegInference:
    def __init__(
        self,
        model_path: Union[Path, str],
        dataset_path: Union[Path, str],
        outdir: Union[Path, str],
        gpu: int,
        patching: bool = False,
        overlap: int = 0,
        magnification: int = 40,
    ) -> None:
```

**Arguments:**
- `model_path`: Path to CellViT model checkpoint (.pth file)
- `dataset_path`: Path to MoNuSeg dataset folder
- `outdir`: Where to save results
- `gpu`: CUDA GPU ID to use
- `patching`: Whether to split images into 256×256 patches
- `overlap`: Overlap between patches (in pixels, e.g., 64)
- `magnification`: Image magnification level (20 or 40×)

**Patching Modes:**

1. **No Patching** (`patching=False`):
   - Process entire image at once
   - Good for images ≤ 256×256
   - Faster but uses more memory

2. **Patching without Overlap** (`patching=True, overlap=0`):
   - Split image into 256×256 tiles
   - Process each independently
   - Good for large images
   - May have artifacts at tile boundaries

3. **Patching with Overlap** (`patching=True, overlap=64`):
   - Split with 64px overlap between tiles
   - Merge predictions intelligently
   - Best quality but slower
   - Reduces boundary artifacts

### 2. Model Loading (Lines 127-188)

```python
def __load_model(self) -> None:
    """Load model and checkpoint from model_path"""
    model_checkpoint = torch.load(self.model_path)
    model = self.__get_model(model_checkpoint["arch"])
    model.load_state_dict(model_checkpoint["model_state_dict"])
```

**What it does:**
- Loads full CellViT segmentation model (not classifier head!)
- Supports CellViT, CellViT256, CellViTSAM architectures
- No classifier loading (binary segmentation only)

**Supported Architectures:**
```python
def __get_model(self, model_type: str):
    if model_type == "CellViT":
        return CellViT(num_nuclei_classes=1, ...)
    elif model_type == "CellViT256":
        return CellViT256(num_nuclei_classes=1, ...)
    elif model_type == "CellViTSAM":
        return CellViTSAM(num_nuclei_classes=1, ...)
```

**Key:** `num_nuclei_classes=1` because it's **binary segmentation**!

### 3. Inference Workflow (Lines 207-266)

The main inference loop:

```python
def run_inference(self, generate_plots: bool = False) -> None:
    for idx, batch in enumerate(dataloader):
        if self.patching:
            if self.overlap > 0:
                predictions = self.inference_step_overlap(batch)
            else:
                predictions = self.inference_step(batch)
        else:
            predictions = self.inference_step(batch)
```

**Three Processing Paths:**

#### Path 1: No Patching (Lines 268-353)
```python
def inference_step(self, batch, patching=False):
    # Process entire image or single patch
    predictions = model(batch["image"])
    # Post-process to get instances
    predictions = calculate_instances(predictions)
    return predictions
```

#### Path 2: Patching without Overlap (Lines 508-551)
```python
def post_process_patching(self, predictions):
    # Split predictions back into original patches
    # Each 256×256 patch processed separately
    # Merge using grid layout
```

#### Path 3: Patching with Overlap (Lines 553-641)
```python
def post_process_patching_overlap(self, predictions, overlap):
    # Complex merging strategy
    # Use edge detection to decide which patch wins
    # Handles overlap regions intelligently
```

### 4. Metrics Calculated (Lines 355-449 and 643-758)

MoNuSeg calculates **binary segmentation metrics** only:

#### Detection Metrics
```python
# Lines 359-377
scores = cell_detection_scores(gt_instance_map, pred_instance_map)
```

**Metrics:**
- **True Positives**: Correctly detected nuclei
- **False Positives**: Detected but not in ground truth
- **False Negatives**: Missed nuclei in ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

#### Segmentation Quality Metrics
```python
# Lines 379-398
pq, dq, sq = get_fast_pq(gt_instance_map, pred_instance_map)
dice_score = dice(pred_binary, gt_binary)
jaccard_score = binary_jaccard_index(pred_binary, gt_binary)
```

**Metrics:**
- **PQ (Panoptic Quality)**: Overall quality (no type distinction needed)
- **DQ (Detection Quality)**: Instance matching quality
- **SQ (Segmentation Quality)**: IoU of matched instances
- **Dice Score**: Overlap coefficient
- **Jaccard Index (IoU)**: Intersection over union

**Note:** These are **binary** metrics - they don't consider cell types!

### 5. Patching and Overlap Explained

#### Why Patching?

MoNuSeg images can be large (e.g., 1000×1000 pixels). Patching helps:
1. **Memory**: Process large images without OOM errors
2. **Efficiency**: Leverage batch processing
3. **Compatibility**: Model trained on 256×256 patches

#### Overlap Strategy

```python
# Without overlap (patching=True, overlap=0)
Image (1024×1024) → 16 patches (256×256) → Process → Merge
Problems: Boundary artifacts, cells cut in half

# With overlap (patching=True, overlap=64)
Image (1024×1024) → Overlapping patches → Process → Smart merge
Benefits: Better boundaries, fewer artifacts
```

**Smart Merge Algorithm (Lines 582-641):**
```python
def merge_predictions(self, predictions, overlap):
    # For each overlapping region:
    # 1. Check which patch has more "edge" pixels (likely incomplete cells)
    # 2. Keep predictions from patch with fewer edges
    # 3. This favors central regions over edges
```

**Recommendation:**
- Small images (≤512×512): No patching
- Medium images (512×1024): Patching, no overlap
- Large images (>1024×1024): Patching with overlap=64

### 6. Output Format (Lines 402-448)

Results saved to `{outdir}/results.json`:

```json
{
  "image_001": {
    "Dice": 0.8234,
    "Jaccard": 0.7543,
    "PQ": 0.6789,
    "DQ": 0.7234,
    "SQ": 0.8456,
    "True_Positives": 45,
    "False_Positives": 3,
    "False_Negatives": 5,
    "F1_Score": 0.8901,
    "Precision": 0.9375,
    "Recall": 0.9000
  },
  // ... for each image
  "summary": {
    "Mean_Dice": 0.8156,
    "Mean_Jaccard": 0.7432,
    "Mean_PQ": 0.6543,
    "Mean_F1": 0.8765,
    // ... average metrics
  }
}
```

## Comparison: When to Use Which Script

| Your Task | Use This Script |
|-----------|----------------|
| Binary nuclei segmentation (MoNuSeg) | ✅ `inference_cellvit_experiment_monuseg.py` |
| Multi-class nuclei (CoNSeP) | `inference_cellvit_experiment_consep.py` |
| Custom multi-class nuclei | `inference_cellvit_experiment_segmentation.py` |
| Tissue + nuclei types (PanNuke) | `inference_cellvit_experiment_pannuke.py` |
| Detection (CSV coordinates) | `inference_cellvit_experiment_detection.py` |

## Usage Examples

### Example 1: Basic Binary Segmentation
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_monuseg.py \
  --model_path ./models/cellvit_monuseg.pth \
  --dataset_path ./data/monuseg \
  --outdir ./results/monuseg \
  --gpu 0 \
  --magnification 40
```

### Example 2: With Patching (No Overlap)
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_monuseg.py \
  --model_path ./models/cellvit_monuseg.pth \
  --dataset_path ./data/monuseg \
  --outdir ./results/monuseg_patched \
  --gpu 0 \
  --patching \
  --magnification 40
```

### Example 3: With Patching and Overlap (Best Quality)
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_monuseg.py \
  --model_path ./models/cellvit_monuseg.pth \
  --dataset_path ./data/monuseg \
  --outdir ./results/monuseg_overlap \
  --gpu 0 \
  --patching \
  --overlap 64 \
  --magnification 40
```

### Example 4: Generate Visualization Plots
```bash
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_monuseg.py \
  --model_path ./models/cellvit_monuseg.pth \
  --dataset_path ./data/monuseg \
  --outdir ./results/monuseg_plots \
  --gpu 0 \
  --generate_plots  # Save visualization images
```

## Common Issues

### Issue 1: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:** Use patching
```bash
--patching  # Split into 256×256 patches
```

### Issue 2: Boundary Artifacts
```
Cells are cut off at patch boundaries
```

**Solution:** Use overlap
```bash
--patching --overlap 64  # 64px overlap between patches
```

### Issue 3: Wrong Model Type
```
Error: num_nuclei_classes mismatch, expected 1
```

**Cause:** Loaded a multi-class model instead of binary

**Solution:** Ensure model was trained for binary segmentation (num_nuclei_classes=1)

### Issue 4: Magnification Mismatch
```
Warning: Model trained at 40×, but evaluating at 20×
```

**Solution:** Match magnifications
```bash
--magnification 40  # Must match training magnification
```

## Performance Considerations

| Configuration | Speed | Quality | Memory | Use Case |
|--------------|-------|---------|--------|----------|
| No patching | Fastest | Best | High | Small images (≤512×512) |
| Patching, no overlap | Fast | Good | Low | Medium images (512×1024) |
| Patching + overlap=64 | Slow | Best | Low | Large images (>1024) |

**Recommendations:**
- **Development/Testing:** No patching (faster iteration)
- **Production:** Patching with overlap (best quality)
- **Benchmarking:** Match original paper settings

## Key Takeaways

**This script is for:**
- ✅ Binary nuclei segmentation (no classification)
- ✅ MoNuSeg benchmark evaluation
- ✅ Large images requiring patching
- ✅ Direct CellViT model evaluation (no classifier head)

**This script is NOT for:**
- ❌ Multi-class nuclei classification
- ❌ Custom classifiers trained with train_cell_classifier_head.py
- ❌ Detection datasets (CSV coordinates)
- ❌ Tissue-type classification

**If you trained a classifier for types, use:**
- `inference_cellvit_experiment_segmentation.py` (generic)
- `inference_cellvit_experiment_consep.py` (CoNSeP benchmark)
- `inference_cellvit_experiment_pannuke.py` (PanNuke benchmark)

## See Also

- `docs/EVALUATION_GUIDE.md` - Comprehensive evaluation guide
- `docs/UNDERSTANDING_CONSEP_SCRIPT.md` - Similar guide for CoNSeP
- `docs/UNDERSTANDING_PANNUKE_SCRIPT.md` - Similar guide for PanNuke
- `docs/COMPARISON_INFERENCE_SCRIPTS.md` - Compare all inference scripts
- `docs/TERMINOLOGY_GUIDE.md` - Understand dataset terminology
