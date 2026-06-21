# Tutorial: Understanding STHELAR and CellViT_for_STHELAR

This document explains the two STHELAR-related repositories and their modifications
relative to the original CellViT/CellViT++ codebase.

## Table of Contents

1. [Overview](#overview)
2. [MICS-Lab/STHELAR — Dataset Pipeline](#mics-labsthelar--dataset-pipeline)
3. [MICS-Lab/CellViT_for_STHELAR — Model Adaptation](#mics-labcellvit_for_sthelar--model-adaptation)
4. [Line-by-Line: Key Modifications](#line-by-line-key-modifications)
5. [Relationship to CellViT++](#relationship-to-cellvit)

---

## Overview

**STHELAR** (Spatial Transcriptomics and H&E Linked Annotations Resource) is a large-scale
multi-tissue dataset that links 10x Genomics Xenium FFPE spatial transcriptomics data with
co-registered H&E whole-slide images.

- **Scale**: ~11M+ annotated cells, 31 slides, 16 tissue types, 22 cancerous + 9 non-cancerous patients
- **Publication**: Scientific Data, 2026
- **Dataset DOI (40x)**: 10.57967/hf/6008
- **Dataset DOI (20x)**: 10.57967/hf/6009
- **BioImage Archive**: S-BIAD2146

Two repositories were created:

| Repository | Purpose |
|------------|---------|
| [MICS-Lab/STHELAR](https://github.com/MICS-Lab/STHELAR) | Full dataset creation pipeline: from raw Xenium data to annotated H&E patches |
| [MICS-Lab/CellViT_for_STHELAR](https://github.com/MICS-Lab/CellViT_for_STHELAR) | Fork of CellViT adapted to train on STHELAR data via HuggingFace format |

---

## MICS-Lab/STHELAR — Dataset Pipeline

### Repository Structure

```
MICS-Lab/STHELAR/
├── src/
│   ├── _1_get_sdata/           # Stage 1: Build SpatialData objects
│   │   └── process_slide.py    # Load Xenium + CellPose segmentation + H&E alignment
│   ├── _2_annotations/         # Stage 2: Cell-type annotation
│   │   ├── _2-2_add_tangram_ref_annots.py  # Tangram deconvolution
│   │   ├── _2-5_add_leiden.py              # Leiden clustering
│   │   ├── _2-8_train_scVI.py             # scVI latent space
│   │   └── ...                             # 11 annotation scripts total
│   ├── _3_check_alignment/     # Stage 3: H&E alignment quality check
│   │   ├── _3-1_build_patches.py           # Build PanNuke-format patches
│   │   └── _3-3_add_align.py              # Compute Dice/Jaccard/bPQ metrics
│   ├── _4_wsi_features/        # Stage 4: WSI feature extraction
│   │   └── 4-1_extract_cells_features.py   # CellViT embeddings (1280-d)
│   ├── _5_build_final_dataset/ # Stage 5: Build final training dataset
│   │   ├── _5-3_get_ds_per_slide_CAT.py   # PanNuke-format patches + type maps
│   │   └── shortcut_build_dataset.py      # Direct CellViT format output
│   └── _6_analyze_trained_model/ # Stage 6: Analysis of trained models
├── data/
│   └── slides_list.txt          # All 31 slide identifiers
└── README.md
```

### Stage 1: SpatialData Ingestion (`process_slide.py`)

This script converts raw 10x Xenium FFPE data into SpatialData `.zarr` format:

```python
# Key operations:
# 1. Load Xenium data using spatialdata_io
sdata = spatialdata_io.xenium(raw_data_path)

# 2. Run CellPose nucleus segmentation on DAPI channel
#    Parameters: diameter=30, flow_threshold=2, cellprob_threshold=-6, min_area=400
#    Patching: patch_width=1200, overlap=50

# 3. Aggregate transcripts per nucleus
aggregator = Aggregator(...)
aggregator.aggregate()

# 4. Load and align H&E .ome.tif using 10x alignment CSV
sopa.io.align(sdata, he_path, alignment_csv)

# 5. Save as zarr
sdata.write(f"sdata_{slide_id}.zarr")
```

### Stage 2: Cell-Type Annotation

The annotation pipeline uses multiple complementary methods:
- **Tangram** deconvolution: maps single-cell reference atlas → Xenium nuclei
- **Leiden clustering**: per-slide community detection to refine annotations
- **scVI**: variational autoencoder for additional latent space validation
- **SingleR** (R-based): orthogonal reference-based annotation
- **MFA**: Multiple Factor Analysis on cytoplasm + nucleus RNA

### Stage 3: H&E Alignment Quality Check

Builds 256×256 H&E patches from the aligned data, runs pre-trained CellViT (segmentation
only) on them, then computes per-patch quality metrics (Dice, Jaccard, Panoptic Quality)
comparing Xenium segmentation masks vs. CellViT predictions.

### Stage 5: Final Dataset Build

`_5-3_get_ds_per_slide_CAT.py` creates the final training data:
- **Output format**: PanNuke-compatible (256×256 H&E patches with instance + type maps)
- **Label space**: 10 cell-type categories derived from Tangram + Leiden consensus
- **Alternative**: `shortcut_build_dataset.py` directly targets CellViT format

> **Important Note**: The "Other" category in the manuscript corresponds to "Dead" in the
> code. The codebase uses "Dead" as an unverified hypothesis; the paper uses the neutral
> term "Other."

---

## MICS-Lab/CellViT_for_STHELAR — Model Adaptation

### Repository Structure (vs. Original CellViT)

```
MICS-Lab/CellViT_for_STHELAR/
├── preprocessing/sthelar/           # ★ NEW: HuggingFace → CellViT converter
│   ├── convert_hf_to_cellvit.py     # Main conversion script (~1300 lines)
│   ├── inspect_dataset.py           # Dataset inspection utility
│   └── visualize_patch.py           # Visualization utility
├── configs/                         # ★ NEW: STHELAR-specific configs
│   ├── training_sthelar.yaml        # Training configuration
│   └── examples/                    # Preprocessing config examples
│       ├── preprocessing_sthelar20x_5class.yaml
│       ├── preprocessing_sthelar20x_9class.yaml
│       └── preprocessing_sthelar20x_cancer_normal.yaml
├── cell_segmentation/               # Modified from original CellViT
│   └── run_cellvit.py               # ★ MODIFIED: Added 'sthelar' dataset support
├── ruche/, jeanzay/                 # ★ NEW: SLURM HPC cluster scripts
└── README_CellViT.md               # Original CellViT README preserved
```

### Key Design Decision

Rather than creating a new experiment class, STHELAR data is **converted to PanNuke format**
and processed through the existing `ExperimentCellVitPanNuke` class. This minimizes code
changes while leveraging the full existing training pipeline.

---

## Line-by-Line: Key Modifications

### Modification 1: Dataset Dispatcher (`run_cellvit.py`)

```python
# ORIGINAL CellViT:
if dataset_name == "pannuke":
    experiment_class = ExperimentCellVitPanNuke

# MODIFIED for STHELAR:
if dataset_name in ["pannuke", "sthelar"]:       # ← Added "sthelar" alias
    experiment_class = ExperimentCellVitPanNuke

# Also added at top of file for Apple Silicon compatibility:
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # ← NEW: MPS support
```

**Why**: STHELAR data is pre-converted to PanNuke format, so it can reuse the same
experiment class. The "sthelar" alias is just for clarity in config files.

---

### Modification 2: HuggingFace-to-CellViT Converter (`convert_hf_to_cellvit.py`)

This is the primary new module (~1300 lines). Here are the key sections:

#### 2a. Label Space Definitions (Lines ~75-93)

```python
# 5-class grouping (maps 9 STHELAR cell types → 4 classes + background)
FIVE_CLASS_MAP = {
    "T_NK": "Immune",
    "B_Plasma": "Immune",
    "Myeloid": "Immune",
    "Blood_vessel": "Stromal",
    "Fibroblast_Myofibroblast": "Stromal",
    "Epithelial": "Epithelial",
    "Melanocyte": "Other",
    "Specialized": "Other",
    "Other": "Other",
}
# PanNuke-compatible type dictionary
NUCLEI_TYPES = {
    "Background": 0,
    "Immune": 1,
    "Stromal": 2,
    "Epithelial": 3,
    "Other": 4,
}
```

**Why**: CellViT's PanNuke experiment expects integer class labels starting at 0
(Background). The grouping reduces 9 fine-grained types to 5 classes that are more
balanced for training.

---

#### 2b. Sparse Cell-ID Map Decoding (Lines ~400-435)

```python
def decode_cell_id_map(npz_bytes: bytes) -> np.ndarray:
    """Decode sparse CSR matrix from .npz bytes to dense instance map."""
    import io
    from scipy.sparse import load_npz

    buf = io.BytesIO(npz_bytes)
    sparse_mat = load_npz(buf)           # scipy CSR matrix
    dense = sparse_mat.toarray()         # → (H, W) int32 array
    return dense.astype(np.int32)
```

**Why**: HuggingFace stores per-pixel cell instance IDs as compressed sparse matrices
(most pixels are background=0). This is space-efficient for parquet storage but must
be densified for CellViT which expects dense numpy arrays.

---

#### 2c. Type Map Construction (Lines ~439-489)

```python
def build_type_map(cell_id_map, slide_meta, label_column,
                   label_mapping, class_to_int, ignore_labels, fallback_class):
    """Convert instance IDs → per-pixel type labels using metadata lookup."""

    # 1. Get unique cell IDs in this patch
    ids_all, inv = np.unique(cell_id_map, return_inverse=True)

    # 2. Look up cell type for each unique ID from metadata
    labels_for_ids = slide_meta[label_column].reindex(ids_all).to_numpy()

    # 3. Map raw labels → target classes via label_mapping
    class_names = []
    for cell_id, raw_label in zip(ids_all, labels_for_ids):
        if cell_id == 0:
            class_names.append("Background")
        elif raw_label in ignore_labels:
            class_names.append("Background")  # Zero out ignored cells
        elif pd.isna(raw_label):
            class_names.append(fallback_class)
        else:
            class_names.append(label_mapping.get(raw_label, fallback_class))

    # 4. Build lookup table: unique_id_index → class_int
    lut = np.array([class_to_int[name] for name in class_names], dtype=np.uint8)

    # 5. Apply LUT to create dense type map (vectorized, very fast)
    type_map = lut[inv].reshape(cell_id_map.shape)

    return type_map
```

**Why**: This is the core transformation. Each pixel in `cell_id_map` has a cell instance
ID. The metadata table maps each cell ID to a cell-type label. This function performs a
vectorized lookup to create a per-pixel class label map, matching PanNuke format.

---

#### 2d. Split Strategies (Lines ~587-985)

Four splitting strategies prevent data leakage:

```python
def assign_splits(patch_info, strategy, train_frac, valid_frac, ...):
    if strategy == "baseline":
        # Random patch-level shuffle (risk of spatial leakage)
        indices = np.random.permutation(len(patch_info))
        # Assign first 70% train, next 15% valid, rest test

    elif strategy == "spatial":
        # Coordinate-based split within each slide
        # Uses x- or y-axis position with boundary margin
        b1 = coord_min + train_frac * span
        b2 = coord_min + (train_frac + valid_frac) * span
        # Patches within ±boundary_margin of b1/b2 are DISCARDED
        # This prevents leakage from adjacent patches

    elif strategy == "slide":
        # Whole-slide assignment
        # ≥3 slides: random assignment by fraction
        # 2 slides: slide 0 → train+valid (spatial split), slide 1 → test

    elif strategy == "auto":
        # slide if ≥2 slides selected, spatial if only 1
```

**Why**: Histopathology patches from the same slide are spatially correlated. Naive
random splitting can leak information. The spatial strategy with boundary margins
ensures clean separation between splits.

---

#### 2e. Output Format (Lines ~1000-1100)

```python
# For each patch, save:
# 1. Image: images/{uid}.png  (256×256 RGB H&E patch)
# 2. Labels: labels/{uid}.npz containing:
#    - 'inst_map': instance segmentation map (H×W, int32)
#    - 'type_map': per-pixel type labels (H×W, uint8)

# Also generate:
# - images.zip, labels.zip     (zipped for easy distribution)
# - types.csv                   (class name → int mapping)
# - cell_count_{split}.csv     (per-patch cell counts by type)
# - dataset_config.yaml         (full configuration record)
# - split_manifest.yaml         (which patches in which split)
```

**Why**: This exactly matches the format expected by CellViT's PanNuke experiment class,
so no further code modifications are needed for training.

---

### Modification 3: Training Configuration (`training_sthelar.yaml`)

```yaml
model:
  backbone: "SAM-H"                    # SAM ViT-H encoder (1280-d embeddings)
  pretrained: "models/pretrained/CellViT-SAM-H-x20.pth"
  embed_dim: 1280
  depth: 32
  num_heads: 16
  extract_layers: 4
  shared_decoders: False               # Separate decoder heads per output

data:
  dataset: "PanNuke"                   # ← STHELAR uses PanNuke format!
  num_nuclei_classes: 5                # Background + 4 classes (5-class setup)
  num_tissue_classes: 1                # Single synthetic tissue
  input_shape: 256                     # 256×256 patches

training:
  optimizer: "AdamW"
  optimizer_hyperparameter:
    lr: 0.0001
    betas: [0.85, 0.85]
  sampling_strategy: "cell"            # Cell-count-weighted sampling
  sampling_gamma: 0.85                 # Smoothing for class imbalance

transformations:                       # 11 augmentations, all p=0.5
  randomrotate90: {p: 0.5}
  horizontalflip: {p: 0.5}
  verticalflip: {p: 0.5}
  downscale: {p: 0.5}
  blur: {p: 0.5}
  gaussnoise: {p: 0.5}
  colorjitter: {p: 0.5, scale_setting: 0.25, scale_color: 0.1}
  superpixels: {p: 0.5}
  zoomblur: {p: 0.5}
  randomsizedcrop: {p: 0.5}
  elastictransform: {p: 0.5}
  normalize: {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}
```

**Key differences from standard CellViT++ configs**:
- Higher augmentation probability (0.5 vs. 0.2) — larger dataset benefits from stronger augmentation
- `sampling_strategy: "cell"` with `sampling_gamma: 0.85` — handles the severe class imbalance in STHELAR
- `shared_decoders: False` — separate decoders for segmentation and classification heads
- Elastic transforms added (not in default configs)

---

## Relationship to CellViT++

### What CellViT_for_STHELAR Does NOT Change

1. **Model architecture** — Same CellViT encoder-decoder with SAM-H backbone
2. **Training loop** — Same `ExperimentCellVitPanNuke` class
3. **Loss functions** — Same combination of segmentation + classification losses
4. **Postprocessing** — Same cell detection post-processing pipeline
5. **Inference** — Same `detect_cells.py` inference pipeline

### What It DOES Change

1. **Data ingestion** — New HuggingFace parquet → PanNuke converter
2. **Label space** — New cell-type definitions (Immune, Stromal, Epithelial, Other)
3. **Split strategy** — Spatial-aware splitting to prevent data leakage
4. **HPC scripts** — SLURM job submission for large-scale training
5. **MPS support** — Apple Silicon GPU fallback environment variable

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   CellViT_for_STHELAR                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  HuggingFace Parquet ──→ convert_hf_to_cellvit.py       │
│       │                        │                         │
│       │                        ▼                         │
│       │                  PanNuke Format                   │
│       │                  (images/ + labels/)              │
│       │                        │                         │
│       ▼                        ▼                         │
│  cell_metadata.parquet   run_cellvit.py                  │
│  (cell type labels)      (dataset="sthelar"|"pannuke")   │
│                                │                         │
│                                ▼                         │
│                    ExperimentCellVitPanNuke               │
│                    (UNCHANGED from CellViT)              │
│                                │                         │
│                                ▼                         │
│                    Fine-tuned CellViT-SAM-H              │
│                    (STHELAR cell types)                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Summary of Modifications

| File | Type | Description |
|------|------|-------------|
| `preprocessing/sthelar/convert_hf_to_cellvit.py` | NEW | HuggingFace → CellViT format converter (1300 lines) |
| `preprocessing/sthelar/inspect_dataset.py` | NEW | Dataset inspection utility |
| `preprocessing/sthelar/visualize_patch.py` | NEW | Patch visualization utility |
| `configs/training_sthelar.yaml` | NEW | Training configuration |
| `configs/examples/*.yaml` | NEW | Preprocessing config examples (5-class, 9-class, cancer/normal) |
| `cell_segmentation/run_cellvit.py` | MODIFIED | Added "sthelar" dataset alias + MPS support |
| `ruche/`, `jeanzay/` | NEW | SLURM HPC scripts |

**Total new code**: ~1500 lines (primarily the converter)
**Modified existing code**: ~3 lines (dataset alias in dispatcher)

This minimal-modification approach demonstrates that CellViT's architecture is sufficiently
general to handle new datasets with only a preprocessing adapter, validating the
PanNuke-format design decision in the original codebase.
