# Comparison of inference_cellvit_experiment_*.py Scripts

This document provides a comprehensive comparison of all `inference_cellvit_experiment_*.py` evaluation scripts in the CellViT++ repository.

## Overview

There are **14 inference scripts** designed for different datasets and evaluation scenarios. They share common patterns but have dataset-specific customizations.

---

## Script Categories

### 1. Base/Parent Classes

**`inference_cellvit_experiment_classifier.py`** (832 lines)
- **Purpose:** Abstract base class for classifier-based inference
- **Inheritance:** ABC (Abstract Base Class)
- **Key Features:**
  - CellViT model loading
  - Classifier head loading
  - Common validation methods
  - Path and dataset validation
  - Logger setup
  - AMP (Automatic Mixed Precision) support
  
**Key Methods:**
- `__init__()` - Initialization
- `_load_cellvit_model()` - Load pretrained CellViT
- `_load_model()` - Load classifier head
- `_validate_paths()` - Validate all required paths
- `_get_cellvit_result()` - Extract cells using CellViT
- `_apply_softmax_reorder()` - Apply softmax and reorder predictions

---

### 2. Segmentation-Based Scripts (with instance masks)

These scripts work with datasets that have instance segmentation annotations (NumPy masks).

#### **`inference_cellvit_experiment_pannuke.py`** (1256 lines)
- **Dataset:** PanNuke (tissue classification + nuclei types)
- **Inheritance:** None (standalone implementation)
- **Unique Features:**
  - Handles tissue types AND nuclei types
  - Direct inference without classifier head
  - Tissue-specific metrics
  - Complex multi-class evaluation

**Key Methods:**
- `run_patch_inference()` - Patch-based inference
- `calculate_step_metric()` - Per-step metrics
- `unpack_masks()` - Process mask annotations

#### **`inference_cellvit_experiment_consep.py`** (839 lines)
- **Dataset:** CoNSeP
- **Inheritance:** CellViTClassifierInferenceExperiment
- **Unique Features:**
  - Binary segmentation focus
  - Simple nuclei classification (no tissue types)
  - MAT file support for ground truth

**Key Methods:**
- `_load_gt_mat()` - Load .mat ground truth
- `_load_gt_npy()` - Load .npy ground truth
- `_calculate_pipeline_scores()` - End-to-end metrics

#### **`inference_cellvit_experiment_segmentation.py`** (1138 lines) ⭐ **NEW**
- **Dataset:** Generic nuclei-only segmentation datasets
- **Inheritance:** CellViTClassifierInferenceExperiment
- **Unique Features:**
  - **Generic/flexible** - works with any nuclei-only dataset
  - No tissue type dependencies
  - Dependency checking before imports
  - Dataset structure validation
  - Auto-detects label folder names

**Key Methods:**
- `check_dependencies()` - Validate Python packages ⭐
- `_validate_dataset_structure()` - Validate dataset layout
- `_load_gt_npy()` - Load NumPy annotations
- `_get_global_classifier_scores()` - Classification metrics
- `_calculate_pipeline_scores()` - Pipeline metrics

**Similar to:** CoNSeP script but more generic and flexible

#### **`inference_cellvit_experiment_monuseg.py`**
- **Dataset:** MoNuSeg (binary segmentation)
- **Inheritance:** Similar to CoNSeP

#### **`inference_cellvit_experiment_nucls.py`**
- **Dataset:** NuCLS (detailed nuclei classification)

---

### 3. Detection-Based Scripts (with coordinate annotations)

These scripts work with datasets that have cell coordinate annotations (CSV format).

#### **`inference_cellvit_experiment_detection.py`** (966 lines)
- **Dataset:** Generic detection datasets (CSV coordinates)
- **Inheritance:** CellViTClassifierInferenceExperiment
- **Unique Features:**
  - Works with CSV-based annotations
  - Coordinate matching (pairing detected cells with ground truth)
  - Per-class F1 scores ⭐
  - Classification-only vs pipeline metrics

**Key Methods:**
- `_get_global_classifier_scores()` - Classification metrics with per-class breakdown ⭐
- `_calculate_pipeline_scores()` - Detection + classification combined
- `_extract_tokens()` - Extract cell features

#### **`inference_cellvit_experiment_lizard.py`** (805 lines)
- **Dataset:** Lizard dataset
- **Inheritance:** CellViTClassifierInferenceExperiment
- **Unique Features:**
  - Lizard-specific data format
  - Histomics variant support

#### **`inference_cellvit_experiment_ocelot.py`**
- **Dataset:** OCELOT challenge

#### **`inference_cellvit_experiment_midog.py`**
- **Dataset:** MIDOG challenge (mitosis detection)

---

### 4. Specialized/Challenge Scripts

#### **`inference_cellvit_experiment_segpath.py`**
- **Dataset:** SegPath challenge

#### **`inference_cellvit_experiment_segpath_nucls.py`**
- **Dataset:** SegPath + NuCLS combined

#### **`inference_cellvit_experiment_panoptils.py`**
- **Dataset:** Panoptic segmentation

#### **`inference_cellvit_experiment_lizard_pycaret.py`**
- **Dataset:** Lizard with PyCaret ML integration

---

## Similarity Analysis

### Common Patterns (Shared by Most Scripts)

1. **Inheritance Structure**
   - Most inherit from `CellViTClassifierInferenceExperiment`
   - Exception: `pannuke.py` is standalone

2. **Core Methods** (Present in most scripts)
   - `__init__()` - Initialization
   - `_load_dataset()` - Dataset loading
   - `_calculate_pipeline_scores()` - Metric calculation
   - `update_cell_dict_with_predictions()` - Add predictions to cells
   - `parse_arguments()` - CLI argument parsing

3. **Common Workflow**
   ```python
   1. Load CellViT model
   2. Load classifier head
   3. Load dataset
   4. For each image:
      a. Extract cells with CellViT
      b. Classify cells with classifier
      c. Calculate metrics
   5. Aggregate and save results
   ```

4. **Metrics Calculated**
   - F1 score (global and per-class)
   - Precision
   - Recall
   - PQ (Panoptic Quality)
   - DQ (Detection Quality)
   - SQ (Segmentation Quality)

---

## Key Differences

### 1. Dataset Format

| Script | Format | Ground Truth |
|--------|--------|-------------|
| detection.py | CSV coordinates | x, y, class |
| segmentation.py | NumPy masks | instance + type maps |
| pannuke.py | NumPy masks | instance + type + tissue |
| consep.py | .mat or .npy | instance masks |

### 2. Classification Scope

| Script | Classifier Types | Notes |
|--------|-----------------|-------|
| detection.py | Custom classes | User-defined types |
| segmentation.py | Custom classes | Nuclei-only (no tissue) |
| pannuke.py | Tissue + Nuclei | Both levels |
| consep.py | Few nuclei types | Binary or few classes |

### 3. Evaluation Metrics

| Script | Metrics | Special Features |
|--------|---------|-----------------|
| detection.py | Classification-only + Pipeline | Per-class F1 ⭐ |
| segmentation.py | Binary + Multi-class PQ | Per-class F1 ⭐ |
| pannuke.py | Tissue-specific PQ | Multi-level evaluation |
| consep.py | Binary segmentation | Simple metrics |

### 4. New Features (Our Additions)

| Feature | Scripts with Feature |
|---------|---------------------|
| Dependency checking | segmentation.py ⭐ |
| Path validation | classifier.py (base) ⭐ |
| Dataset validation | segmentation.py ⭐ |
| Per-class F1 scores | detection.py, segmentation.py ⭐ |
| Checkpoint selection | All (via base class) ⭐ |

---

## Code Duplication Analysis

### High Duplication (>80% similar)

1. **consep.py ↔ segmentation.py** (~85% similar)
   - Both handle nuclei-only segmentation
   - Difference: segmentation.py is more generic/flexible
   - Could potentially merge with flag for dataset type

2. **monuseg.py ↔ consep.py** (~90% similar)
   - Both handle binary/simple segmentation
   - Very similar structure

### Moderate Duplication (50-80% similar)

1. **detection.py ↔ lizard.py** (~60% similar)
   - Both use coordinate-based matching
   - Different data loading methods

2. **All subclasses ↔ classifier.py** (~50% similar)
   - All inherit common methods
   - Dataset-specific customizations

### Low Duplication (<50% similar)

1. **pannuke.py ↔ others** (~30% similar)
   - Standalone implementation
   - Different architecture

---

## Recommendations

### For Users

**Which script to use:**

1. **CSV annotations (x, y, class)** → `inference_cellvit_experiment_detection.py`
2. **NumPy masks, nuclei-only** → `inference_cellvit_experiment_segmentation.py` ⭐ **RECOMMENDED**
3. **NumPy masks, tissue+nuclei** → `inference_cellvit_experiment_pannuke.py`
4. **Specific benchmark dataset** → Use dataset-specific script

### For Developers

**Code Improvement Opportunities:**

1. **Reduce Duplication**
   - Extract common metric calculation to shared utility
   - Create base dataset loader classes
   - Standardize metric output format

2. **Improve Maintainability**
   - Apply bug fixes to ALL scripts (e.g., logger.info() bug)
   - Standardize error handling patterns
   - Add dependency checking to all scripts

3. **Enhance Flexibility**
   - Make more scripts generic like segmentation.py
   - Add configuration files instead of hardcoded values
   - Support multiple ground truth formats in one script

---

## Bug Fixes Applied (This PR)

### Fixed in Multiple Scripts

1. **Logger Bug** - `logger.info(load_state_dict())` ⭐⭐⭐
   - Fixed in: `inference_cellvit_experiment_classifier.py` (lines 322, 380)
   - Affects: All scripts inheriting from base class

2. **argmax Empty Sequence** ⭐⭐⭐
   - Fixed in: `cellpostprocessor.py`, `overlap_cell_cleaner.py`
   - Affects: All scripts using postprocessor

3. **nuclei_type_map Shape Mismatch** ⭐⭐⭐
   - Fixed in: `inference_cellvit_experiment_segmentation.py` (line 912)
   - Specific to: Segmentation script only

### Should Be Applied to Other Scripts

1. **Dependency Checking**
   - Currently only in: segmentation.py
   - Should add to: detection.py, pannuke.py, consep.py

2. **Path Validation**
   - Currently in: classifier.py (base)
   - Inherited by: All subclasses ✅

3. **Per-Class F1 Scores**
   - Currently in: detection.py, segmentation.py
   - Could add to: pannuke.py, consep.py

---

## Summary

### Similarities ✅
- All use CellViT for cell detection
- All support classifier heads for cell type classification
- Similar evaluation workflow
- Common metrics (F1, precision, recall, PQ)

### Differences ⚠️
- Dataset format (CSV vs NumPy vs .mat)
- Annotation complexity (coordinates vs masks vs tissue+nuclei)
- Metric granularity (per-class vs global)
- Code structure (inheritance vs standalone)

### Best Practices from Our Work ⭐
1. Dependency checking before imports
2. Path validation before execution
3. Dataset structure validation
4. Per-class metric calculation
5. Clear error messages with actionable guidance
6. Comprehensive documentation

These patterns should be applied to all inference scripts for consistency and robustness.
