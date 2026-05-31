# Dataset Processing Scripts — Line-by-Line Documentation

This document provides a detailed, line-by-line explanation of every script under `cellvit/training/datasets/`. The goal is to help you understand the data-loading pipeline used for training CellViT models on various cell/nuclei segmentation and classification benchmarks.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [`__init__.py`](#__init__py)
3. [`base_cell_dataset.py`](#base_cell_datasetpy)
4. [`dataset_coordinator.py`](#dataset_coordinatorpy)
5. [`pannuke.py`](#pannukepy)
6. [`prepare_pannuke.py`](#prepare_pannukepy)
7. [`monuseg.py`](#monusegpy)
8. [`consep.py`](#conseppy)
9. [`ocelot.py`](#ocelotpy)
10. [`detection_dataset.py`](#detection_datasetpy)
11. [`segmentation_dataset.py`](#segmentation_datasetpy)
12. [`lizard.py`](#lizardpy)
13. [`midog.py`](#midogpy)
14. [`nucls.py`](#nuclspy)
15. [`panoptils.py`](#panoptilspy)
16. [`segpath.py`](#segpathpy)

---

## Architecture Overview

```
CellDataset (abstract base)        BaseCellEmbeddingDataset
     │                                      │
     └── PanNukeDataset                     └── (used for pre-extracted tokens)
         (full segmentation pipeline)

torch.utils.data.Dataset
     ├── MoNuSegDataset          (nuclei segmentation, HV-map generation)
     ├── CoNSePDataset           (cell detection with JSON annotations)
     ├── OcelotDataset           (cell detection with CSV annotations)
     ├── DetectionDataset        (generic detection with CSV annotations)
     ├── SegmentationDataset     (generic segmentation with .npy labels)
     ├── LizardGraphDataset      (pre-extracted graph data)
     ├── LizardHistomicsDataset  (graph data with feature normalization)
     ├── MIDOGDataset            (mitotic figure detection from TIFF WSIs)
     ├── NuCLSDataset            (multi-level cell classification)
     ├── PanoptilsDataset        (TIL detection)
     └── SegPathDataset          (H&E + IHC mask segmentation)
```

All detection-style datasets return the same tuple format:
```
(image_tensor, detections_list, types_list, image_name)
```

The segmentation-style datasets (PanNuke, MoNuSeg) return richer mask dictionaries.

---

## `__init__.py`

```python
# -*- coding: utf-8 -*-
# Datasets
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
```

**Purpose:** Package marker. No imports or logic — merely declares `cellvit/training/datasets/` as a Python package.

---

## `base_cell_dataset.py`

This file defines two abstract/base dataset classes that other datasets inherit from.

### Lines 1–17: Imports and Logger

```python
import logging
from abc import abstractmethod
from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())
```

- Standard library imports for abstract classes and type hints.
- `torch.utils.data.Dataset` is the PyTorch base class for all datasets.
- A module-level logger is created with a `NullHandler` to suppress logs unless a handler is configured upstream.

### Lines 20–85: `CellDataset` (Abstract Base Class)

```python
class CellDataset(Dataset):
```

Inherits from `torch.utils.data.Dataset`. This is the base class for **segmentation** datasets (currently only PanNuke).

#### `set_transforms(self, transforms: Callable)`
Allows dynamically swapping the augmentation pipeline after construction.

#### `load_cell_count(self)` (abstract)
Must be implemented by subclasses. Expected to read a `cell_count.csv` file with per-image cell type counts (used for weighted sampling).

#### `get_sampling_weights_tissue(self, gamma)` (abstract)
Computes per-sample weights based on tissue type distribution. `gamma=1` means full rebalancing; `gamma=0` means uniform weights.

#### `get_sampling_weights_cell(self, gamma)` (abstract)
Computes per-sample weights based on cell type distribution.

#### `get_sampling_weights_cell_tissue(self, gamma)`
Concrete method that:
1. Calls `get_sampling_weights_tissue(gamma)` → `tw`
2. Calls `get_sampling_weights_cell(gamma)` → `cw`
3. Normalizes each to [0, 1] by dividing by their max
4. Returns `tw_norm + cw_norm` as combined weights

### Lines 88–127: `BaseCellEmbeddingDataset`

```python
class BaseCellEmbeddingDataset(Dataset):
```

Used for **classifier-head training** on pre-extracted cell embeddings (tokens).

#### `__init__(self, extracted_cells: List[dict])`
Takes a list of dictionaries, each representing one cell:
- `image`: source image name
- `coords`: (x, y) position
- `type`: ground-truth cell type (int32)
- `token`: embedding tensor of shape `[embed_dim]`

#### `__len__(self)`
Returns the number of cells.

#### `__getitem__(self, index)`
Returns a tuple of:
1. `cell_token` — the pre-extracted embedding
2. `cell_coords` — position as a `[x, y]` tensor
3. `cell_type` — as `torch.long`
4. `image_name` — source image identifier

---

## `dataset_coordinator.py`

A simple factory function that maps dataset names to dataset classes.

### Lines 1–12: Imports

```python
from typing import Callable
from torch.utils.data import Dataset
from cellvit.training.datasets.pannuke import PanNukeDataset
```

Only PanNuke is imported — this coordinator is used for segmentation training only.

### Lines 15–57: `select_dataset(...)`

```python
def select_dataset(dataset_name, split, dataset_config, transforms=None) -> Dataset:
```

**Parameters:**
- `dataset_name`: Must be `"pannuke"` (case-insensitive).
- `split`: One of `"train"`, `"val"`, `"validation"`, `"test"`.
- `dataset_config`: Dict with keys like `dataset_path`, `train_folds`, `val_folds`, `test_folds`, `stardist`, `regression_loss`.
- `transforms`: Optional augmentation pipeline.

**Logic:**
1. Validates `split` against allowed values.
2. Maps split → fold numbers from `dataset_config`.
3. Instantiates `PanNukeDataset` with the resolved folds and options.
4. Raises `NotImplementedError` for any other dataset name.

---

## `pannuke.py`

The most complex dataset class — handles the PanNuke benchmark for panoptic nuclei segmentation.

### Lines 1–32: Imports

Key dependencies:
- `numba.njit` — JIT compilation for the StarDist map generator
- `scipy.ndimage.center_of_mass`, `distance_transform_edt` — for computing HV-maps and distance maps
- `natsort.natsorted` — natural file sorting
- Internal utils: `fix_duplicates`, `get_bounding_box`

### Lines 34–99: `PanNukeDataset.__init__`

```python
class PanNukeDataset(CellDataset):
```

**Constructor parameters:**
| Parameter | Description |
|-----------|-------------|
| `dataset_path` | Root path with `fold0/`, `fold1/`, `fold2/` subdirectories |
| `folds` | Which fold(s) to load (int or list of ints) |
| `transforms` | Albumentations pipeline |
| `stardist` | Whether to generate StarDist vector maps |
| `regression` | Whether to generate regression offset maps |
| `cache_dataset` | Whether to cache images/masks in RAM |

**Initialization steps:**
1. Converts single fold int to list.
2. For each fold, globs `fold{n}/images/*.png`.
3. For each image, checks that `fold{n}/labels/{stem}.npy` exists; if so, adds to `self.images` and `self.masks`.
4. Reads `fold{n}/types.csv` to build a `{image_name: tissue_type}` mapping.
5. Logs dataset length.

### Lines 101–106: Cache setup
If `cache_dataset=True`, initializes empty dicts for indexed caching during the first epoch.

### Lines 108–186: `__getitem__`

Returns: `(image_tensor, masks_dict, tissue_type_str, image_name_str)`

**Step-by-step:**
1. **Load image and mask** — either from cache or disk (`load_imgfile` / `load_maskfile`).
2. **Apply transforms** — albumentations augmentation on image+mask jointly.
3. **Extract maps from mask:**
   - `inst_map = mask[:,:,0]` — instance IDs (channel 0 of the .npy)
   - `type_map = mask[:,:,1]` — cell type per pixel (channel 1)
   - `np_map` — binary nuclei presence (inst_map > 0 → 1)
4. **Generate HV-map** — calls `gen_instance_hv_map(inst_map)`.
5. **Convert image to tensor** — via `ToTensorV2`.
6. **Build masks dict:**
   - `instance_map`: int64 tensor
   - `nuclei_type_map`: int64 tensor
   - `nuclei_binary_map`: int64 binary tensor
   - `hv_map`: float32 tensor of shape (2, H, W)
7. **Optional StarDist:** adds `dist_map` (distance probability) and `stardist_map` (n_rays, H, W).
8. **Optional Regression:** adds `regression_map` (2, H, W).

### Lines 204–230: `load_imgfile` / `load_maskfile`

- `load_imgfile`: Opens PNG as uint8 numpy array (H, W, 3).
- `load_maskfile`: Loads `.npy` dict with `inst_map` and `type_map`, stacks them as (H, W, 2).

### Lines 232–250: `load_cell_count`

Reads `cell_count.csv` from each fold, concatenates into a DataFrame, and reindexes to match `self.img_names` order.

### Lines 252–289: `get_sampling_weights_tissue`

1. Reads `weight_config.yaml` from the dataset root (contains tissue name → count mapping).
2. For each tissue: `w = total_count / (gamma * tissue_count + (1-gamma) * total_count)`.
3. Maps each image to its tissue type weight.

### Lines 291–314: `get_sampling_weights_cell`

1. Uses hardcoded `binary_weight_factors = [4191, 4132, 6140, 232, 1528]` (PanNuke class frequencies for Neoplastic, Inflammatory, Connective, Dead, Epithelial).
2. Clips cell counts to binary (presence/absence).
3. Computes inverse-frequency weight vector.
4. For each image: weighted sum of binary presence indicators.
5. Replaces zero-weight images with the minimum non-zero weight.

### Lines 334–415: `gen_instance_hv_map` (static)

Generates **Horizontal-Vertical (HV) maps** — the core label representation for the HoVer-Net decoder branch.

**Algorithm for each nucleus instance:**
1. Extract bounding box, expand by 2px.
2. Crop the instance mask.
3. Compute center of mass.
4. Create coordinate grids shifted to center of mass.
5. Zero out pixels outside the instance.
6. Normalize negative values to [-1, 0] and positive to [0, 1].
7. Write back to the full-image x_map and y_map.

Returns shape `(2, H, W)` — first channel is horizontal gradient, second is vertical.

### Lines 417–460: `gen_distance_prob_maps` (static)

For each instance:
1. Compute the Euclidean distance transform (distance from boundary).
2. Normalize to [0, 1] by dividing by max distance.
3. Write into the output map.

Returns shape `(H, W)` — a probability-like map where center-of-nucleus ≈ 1.

### Lines 462–509: `gen_stardist_maps` (static, JIT-compiled)

Generates StarDist radial distance vectors with **32 rays**.

**Algorithm (per pixel):**
1. If background (0), output zeros.
2. For each of 32 angles (evenly spaced around 2π):
   - Ray-march outward from the pixel.
   - Stop when leaving the instance or reaching image boundary.
   - Record the distance (with boundary correction).

Returns shape `(32, H, W)`.

### Lines 511–537: `gen_regression_map` (static)

For each instance:
1. Compute center of mass.
2. Create x-distance and y-distance grids from center.
3. Mask to instance pixels.

Returns shape `(2, H, W)` — raw pixel offsets (not normalized).

---

## `prepare_pannuke.py`

A **standalone CLI script** that converts raw PanNuke numpy dumps into per-image PNG + npy files.

### Lines 9–16: Path manipulation
Adds the project root to `sys.path` so internal utilities can be imported.

### Lines 28–69: `process_fold(fold, input_path, output_path)`

For each fold (0, 1, 2):

1. **Load large numpy files:**
   - `images.npy` — shape (N, 256, 256, 3)
   - `masks.npy` — shape (N, 256, 256, 5) where 5 channels = 5 cell type layers

2. **Process images (lines 41–45):**
   - Save each image as `{fold}_{i}.png`.

3. **Process masks (lines 48–69):**
   - For each image, iterate over 5 mask layers (one per cell type).
   - `remap_label()` ensures contiguous instance IDs within each layer.
   - Merge layers into a single `inst_map` by offsetting instance IDs.
   - Final `remap_label()` on the merged map.
   - Build `type_map`: pixel value = (layer_index + 1) where non-zero.
   - Save as `.npy` dict with keys `inst_map` and `type_map`.

### Lines 72–97: CLI interface
Uses argparse with `--input_path` and `--output_path`. Processes all 3 folds.

---

## `monuseg.py`

MoNuSeg dataset for nuclei segmentation evaluation.

### Lines 27–63: `MoNuSegDataset.__init__`

**Parameters:**
- `dataset_path`: Contains `images/` (PNG) and `labels/` (npy) subdirectories.
- `transforms`: Albumentations pipeline.
- `patching`: If True, splits large images into 256×256 patches.
- `overlap`: Overlap in pixels between patches (useful value: 64).

**Initialization:**
1. Globs PNG images and npy masks.
2. Sanity check: verifies each image has a matching mask by stem name.

### Lines 65–112: `__getitem__`

1. Load image (PNG → uint8 array).
2. Load mask (npy → int64 instance map).
3. Apply transforms (joint image+mask augmentation).
4. Generate HV-map using `PanNukeDataset.gen_instance_hv_map(mask)`.
5. Create binary nuclei map.
6. Convert image to tensor, normalize to [0,1] if pixel values > 5.
7. **If patching:** use `einops.rearrange` (no overlap) or `torch.unfold` (with overlap) to split into 256×256 patches.
8. Return `(image, masks_dict, filename)`.

---

## `consep.py`

CoNSeP (Colorectal Nuclear Segmentation and Phenotypes) dataset.

### Lines 27–97: `CoNSePDataset.__init__`

**Parameters:**
- `dataset_path`: Root with `{split}/images/` (PNG) and `{split}/detections/` (JSON).
- `split`: `"Train"` or `"Test"`.
- `filelist_path`: Optional CSV to select a subset of images.
- `transforms`: Default normalizes to mean=0.5, std=0.5.
- `normalize_stains`: Macenko stain normalization.
- `merge_classes`: Merge 7 cell types → 4 classes.

**Cell type mapping (7 classes):**
| ID | Type |
|----|------|
| 0 | Other |
| 1 | Inflammatory |
| 2 | Healthy Epithelial |
| 3 | Dysplastic/Malignant Epithelial |
| 4 | Fibroblast |
| 5 | Muscle |
| 6 | Endothelial |

**Merged mapping (4 classes):**
| Merged ID | Type | Original IDs |
|-----------|------|--------------|
| 0 | Miscellaneous | 0 |
| 1 | Inflammatory | 1 |
| 2 | Epithelium | 2, 3 |
| 3 | Spindle-Shaped | 4, 5, 6 |

### Lines 98–109: `cache_dataset`
Pre-loads all images and JSON annotations into dictionaries keyed by image stem.

### Lines 114–152: `__getitem__`

1. Load image and annotation from cache.
2. Extract `detections` as (x, y) centroids from JSON.
3. Extract `types` as integer labels (subtracts 1 to make 0-indexed).
4. Optional stain normalization via Macenko.
5. Apply albumentations transforms (supports keypoint augmentation).
6. Optionally merge classes using `merged_nuclei_dict`.
7. Return `(image, detections, types, name)`.

### Lines 154–174: `collate_batch` (static)
Custom collate function — stacks images into a batch tensor, keeps detections and types as lists of lists.

---

## `ocelot.py`

OCELOT challenge dataset (2 cell types: Other cells, Tumor cells).

### Lines 25–77: `OcelotDataset.__init__`

Very similar to CoNSeP but:
- Images are JPG (not PNG).
- Annotations are CSV files (not JSON) with format: `x, y, type`.
- Path structure: `images/{split}/cell/` and `annotations/{split}/cell/`.
- Supports train/val/test splits natively.
- Only 2 cell types.

### Lines 79–91: `cache_dataset`
Pre-loads images (PIL) and CSV annotations into memory.

### Lines 96–130: `__getitem__`
Same flow as CoNSeP: load from cache → parse detections/types → stain normalize → augment → return.

**Key difference:** Types are parsed directly from CSV rows `(x, y, type)` with `type - 1` to zero-index.

---

## `detection_dataset.py`

A **generic** detection dataset that works with any dataset following the expected folder structure.

### Expected folder structure:
```
dataset_root/
├── {split}/
│   ├── images/      (*.png, *.jpg, *.jpeg)
│   └── labels/      (*.csv with columns: x, y, type)
```

### Lines 26–81: `DetectionDataset.__init__`

Identical pattern to Ocelot/CoNSeP:
1. Set up paths based on `dataset_path / split / images` and `dataset_path / split / labels`.
2. Glob all image files (supports PNG, JPG, JPEG).
3. Filter by filelist if provided.
4. Map each image to its corresponding CSV label file.

**Key difference from Ocelot:** Types are NOT decremented by 1 — they are used as-is from the CSV.

### Lines 83–95: `cache_dataset`
Pre-loads images and CSV rows into memory.

### Lines 100–134: `__getitem__`
Same standard flow: load → parse → normalize → augment → return.

---

## `segmentation_dataset.py`

A **generic** segmentation dataset that derives cell detections from instance+type maps.

### Expected folder structure:
```
dataset_root/
├── {split}/
│   ├── images/      (*.png, *.jpg, *.jpeg)
│   └── labels/      (*.npy — dict with "inst_map" and "type_map" keys)
```

### Lines 28–83: `SegmentationDataset.__init__`

Same pattern as DetectionDataset but labels are `.npy` files instead of CSVs.

### Lines 85–119: `cache_dataset`

**This is the key differentiator.** Instead of just loading annotations, it **derives cell detections from the segmentation masks:**

For each instance in the instance map:
1. Create a binary mask for that instance.
2. Compute center of mass → (x, y) centroid.
3. Look up the cell type from the type map (majority vote if multiple values).
4. Store as `(x, y, type)` tuple.

This converts a segmentation annotation into a detection annotation on-the-fly.

### Lines 124–158: `__getitem__`
Same as detection datasets — uses the derived `(x, y, type)` annotations. Types are decremented by 1.

---

## `lizard.py`

Two dataset classes for working with **pre-extracted cell graphs** (CellViT inference outputs).

### Lines 20–136: `LizardGraphDataset`

**Purpose:** Loads pre-computed cell graphs for graph-based classification.

**Expected structure:**
```
dataset_root/
├── {split}/
│   ├── labels/                    (*.mat files from original Lizard)
│   └── predictions-cellvit/
│       └── {network_name}/        (e.g., SAM-H, UNI, ViT256)
│           ├── {image}_cells.pt   (CellGraphDataWSI tensor)
│           └── {image}_cells.json (cell metadata dict)
```

**`__init__` (lines 21–81):**
1. Resolves paths for graph outputs and ground-truth annotations.
2. For each `.mat` annotation file, asserts the corresponding `.pt` and `.json` exist.
3. Defines 6 cell types (Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue).

**`__getitem__` (lines 86–120):**
1. `torch.load()` the graph `.pt` file → `CellGraphDataWSI` object.
2. `json.load()` the cell dict (contains per-cell metadata: bbox, centroid, contour, type_prob, type).
3. `loadmat()` the `.mat` ground-truth (centroid coordinates, class labels, instance map).
4. Build `gt_dict` with detections (x,y), types (0-indexed), and inst_map tensor.

**Important note:** Graphs are extracted at 40× magnification but ground-truth is at 20×.

### Lines 139–271: `LizardHistomicsDataset`

Extends `LizardGraphDataset` with **feature normalization** for histomics features.

**Additional `__init__` parameters:**
- `mean`: List of feature means (length = embedding_dim).
- `std`: List of feature stds.

**`__getitem__` additions (lines 220–243):**
1. Replace NaN values in graph features with the corresponding mean.
2. Z-score normalize: `x = (x - mean) / std` (with std=0 → 1 protection).
3. Rescale positions by 2× (converting from 0.5 µm/px to 0.25 µm/px, i.e., 20× → 40×).

---

## `midog.py`

MIDOG dataset for mitotic figure detection from whole-slide TIFF images.

### Lines 31–58: Class docstring and attributes

Key attributes:
- `slide_cache`: Cached TIFF file handles
- `image_crops`: Generated crop regions around cells
- `data_elements`: Flat list of (image_id, crop) pairs for indexing

### Lines 60–134: `MIDOGDataset.__init__`

1. Set up paths: `images/` folder for TIFFs, `midog.json` for COCO-style annotations.
2. Load filelist CSV to select which images to include.
3. Load and parse `midog.json` (COCO format: images + annotations arrays).
4. Build `image_ids` mapping (filename → ID) and `ids_image_paths` (ID → Path).
5. Initialize random number generator with `crop_seed` for reproducibility.
6. Extract image metadata (shape, filename, tumor_type).
7. **`_prepare_dataset()`** — generates crops around annotated cells.
8. Flatten all crops into `data_elements` list (each element = one training sample).

### Lines 143–203: `__getitem__`

1. Get the (image_id, crop) for this index.
2. Load the TIFF image (from cache or disk via `tifffile`).
3. Crop to the specified region.
4. Convert to RGB.
5. Optional stain normalization.
6. Find all cell annotations whose bounding box falls within this crop.
7. Compute centroids from bounding boxes, shift to crop-local coordinates.
8. Apply augmentation transforms.
9. Return `(image, detections, types, image_name)`.

### Lines 252–335: `_prepare_dataset`

The crop generation algorithm:
1. For each annotated cell, check if it's already covered by an existing crop.
2. If not, create a new 1024×1024 crop centered on the cell (with random offset).
3. Ensure crop stays within image bounds; adjust if needed.
4. After all crops generated, **clean** by removing subset crops (crops whose cells are all contained in another crop).

### Lines 337–363: `clean_crops`

Removes redundant crops:
1. Build a mapping: crop → list of cell IDs it contains.
2. If crop A's cells are a strict subset of crop B's cells, remove crop A.

### Lines 365–387: `check_crop_exists` (static)

Simple axis-aligned bounding box containment test.

---

## `nucls.py`

NuCLS dataset supporting **three classification granularities**.

### Lines 26–122: `NuCLSDataset.__init__`

**Special parameter:**
- `classification_level`: One of `"raw_classification"` (12 types), `"main_classification"` (7 types), or `"super_classification"` (4 types).

**Label maps:**

| Level | Classes |
|-------|---------|
| raw | tumor, mitotic_figure, fibroblast, vascular_endothelium, macrophage, lymphocyte, plasma_cell, neutrophil, eosinophil, myoepithelium, apoptotic_body, ductal_epithelium |
| main | tumor_nonMitotic, tumor_mitotic, nonTILnonMQ_stromal, macrophage, lymphocyte, plasma_cell, other_nucleus |
| super | tumor_any, nonTIL_stromal, sTIL, other_nucleus |

An `inverse_label_map` (name → ID) is also built for label lookup.

**File structure:**
```
dataset_root/
├── {split}/
│   ├── images/   (*.png)
│   └── labels/   (*.csv with columns: x, y, raw_classification, main_classification, super_classification)
```

**Filelist filtering note (line 72):** Matches on `f.stem.split("_")[0]` — the prefix before the first underscore.

### Lines 124–135: `cache_dataset`
Reads CSV annotations into pandas DataFrames.

### Lines 140–180: `__getitem__`

1. Read cached image and annotation DataFrame.
2. Parse rows: extract (x, y, label) tuples for the selected classification level.
3. **Filter:** only include cells whose label exists in `inverse_label_map` (skips unknown labels).
4. Map string labels → integer IDs.
5. Standard normalize → augment → return flow.

---

## `panoptils.py`

PanopTILs dataset for tumor-infiltrating lymphocyte detection.

### Lines 24–78: `PanoptilsDataset.__init__`

**4 cell types:**
| ID | Type |
|----|------|
| 0 | Other Cells |
| 1 | Epithelial Cells |
| 2 | Stromal Cells |
| 3 | TILs |

**Structure:**
```
dataset_root/
├── {split}/
│   ├── images/       (*.png)
│   └── annotations/  (*.csv with columns: x, y, type)
```

### Lines 80–81: `cache_dataset`
Logs a warning — dataset is too large to cache.

### Lines 86–125: `__getitem__`

**Key difference from other detection datasets:** Images are loaded fresh from disk each time (no caching), because the dataset is too large.

The CSV format is the same as Ocelot: `x, y, type`. Types are used as-is (not decremented).

---

## `segpath.py`

SegPath dataset for H&E → IHC segmentation.

### Lines 28–67: `SegPathDataset.__init__`

**Parameters:**
- `dataset_path`: Flat directory with `*_HE.png` and `*_mask.png` files.
- `filelist_path`: Optional CSV to filter images.
- `transforms`: Default includes `CenterCrop(960, 960)`.
- `ihc_threshold`: Threshold for binarizing the IHC mask (default 0.2). Stored but not used in `__getitem__` — intended for downstream use.

**`_create_dataset()` (lines 69–85):**
1. If filelist provided: construct paths as `{name}_HE.png`.
2. Otherwise: glob all PNG files containing "HE" in filename.
3. Build annotation dict: `{stem_without_HE: path_to_mask_png}`.

### Lines 98–129: `__getitem__`

1. Load H&E image (PNG → RGB).
2. Load mask image (PNG → numpy array — typically grayscale IHC probability).
3. Optional stain normalization.
4. Apply transforms (jointly on image and mask).
5. Return `(image_tensor, mask_tensor, image_name)`.

**Note:** Unlike detection datasets, this returns a **dense mask** rather than point detections.

### Lines 131–140: `collate_batch`
Stacks both images and masks into batch tensors (both are dense).

---

## Common Patterns Across All Datasets

### 1. Stain Normalization (Macenko)
```python
if self.normalize_stains:
    img = to_tensor(img)                           # PIL → tensor [C, H, W] in [0,1]
    img = (255 * img).type(torch.uint8)            # Scale to [0, 255] uint8
    img, _, _ = self.normalizer.normalize(img)     # Macenko normalization
    img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))  # Back to PIL
```

This standardizes the color distribution across different staining protocols.

### 2. Albumentations with Keypoints
```python
transformed = self.transforms(image=img, keypoints=detections)
img = transformed["image"]
detections = transformed["keypoints"]
types = [types[idx] for idx, _ in enumerate(detections)]
```

Albumentations can augment keypoints consistently with the image. After augmentation, some keypoints may be removed (if they fall outside the image), so the types list is re-synced.

### 3. Custom Collate Functions
All detection datasets use the same pattern:
```python
@staticmethod
def collate_batch(batch):
    imgs, detections_list, types_list, names = zip(*batch)
    imgs = torch.stack(imgs)  # Stack images into batch
    return imgs, list(detections_list), list(types_list), list(names)
```

This is needed because detections have variable length per image (can't be stacked into a tensor).

### 4. Filelist Filtering
```python
if filelist_path is not None:
    selected_files = []
    with open(filelist_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            selected_files.append(row[0])
    self.images = [f for f in self.images if f.stem in selected_files]
```

A simple CSV-based mechanism to select a subset of images (useful for custom train/val splits).

---

## Data Flow Summary

```
Raw Dataset (varies by source)
       │
       ▼  [prepare_pannuke.py for PanNuke]
Processed Dataset on Disk
       │
       ▼  [dataset_coordinator.py selects class]
Dataset Class Instance
       │
       ▼  [cache_dataset() pre-loads to RAM]
Cached Data
       │
       ▼  [__getitem__() called by DataLoader]
       │
       ├── Load image + annotation
       ├── Stain normalization (optional)
       ├── Augmentation (albumentations)
       ├── Generate derived maps (HV, StarDist, etc.) [segmentation only]
       └── Return (image, labels, metadata)
       │
       ▼  [collate_batch() merges samples]
Training Batch
```
