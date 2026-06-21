# -*- coding: utf-8 -*-
# STHELAR Dataset Preprocessing Script
#
# Downloads the STHELAR dataset from HuggingFace and converts it
# to the format expected by CellViT++ classifier training.
#
# Dataset: https://huggingface.co/datasets/FelicieGS/STHELAR_40x
#
# Usage:
#   python -m cellvit.training.datasets.sthelar.prepare_sthelar \
#       --output_dir /path/to/output \
#       --magnification 40x \
#       --num_classes 5 \
#       --split_strategy spatial \
#       --max_patches_per_slide 5000
#
# Output structure (for classifier training):
#   output_dir/
#   ├── train/
#   │   ├── images/       (*.png, 256x256 RGB)
#   │   └── detections/   (*.json, cell centroids + types)
#   ├── val/
#   │   ├── images/
#   │   └── detections/
#   ├── test/
#   │   ├── images/
#   │   └── detections/
#   ├── cell_count_train.csv
#   ├── cell_count_val.csv
#   ├── cell_count_test.csv
#   └── dataset_config.yaml

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================================
# Label Space Definitions
# ==============================================================================

# 5-class grouping: 9 STHELAR cell types → 4 classes + background
FIVE_CLASS_LABEL_MAPPING = {
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

FIVE_CLASS_NUCLEI_TYPES = {
    "Background": 0,
    "Immune": 1,
    "Stromal": 2,
    "Epithelial": 3,
    "Other": 4,
}

# 9-class: identity mapping (one class per STHELAR cell type)
NINE_CLASS_LABEL_MAPPING = {
    "Epithelial": "Epithelial",
    "Blood_vessel": "Blood_vessel",
    "Fibroblast_Myofibroblast": "Fibroblast_Myofibroblast",
    "Myeloid": "Myeloid",
    "B_Plasma": "B_Plasma",
    "T_NK": "T_NK",
    "Melanocyte": "Melanocyte",
    "Specialized": "Specialized",
    "Other": "Other",
}

NINE_CLASS_NUCLEI_TYPES = {
    "Background": 0,
    "Epithelial": 1,
    "Blood_vessel": 2,
    "Fibroblast_Myofibroblast": 3,
    "Myeloid": 4,
    "B_Plasma": 5,
    "T_NK": 6,
    "Melanocyte": 7,
    "Specialized": 8,
    "Other": 9,
}

# 2-class: cancer vs normal
CANCER_NORMAL_LABEL_MAPPING = {
    "cancer_cell": "Cancer",
    "normal_cell": "Normal",
}

CANCER_NORMAL_NUCLEI_TYPES = {
    "Background": 0,
    "Cancer": 1,
    "Normal": 2,
}

LABEL_CONFIGS = {
    5: {
        "label_mapping": FIVE_CLASS_LABEL_MAPPING,
        "nuclei_types": FIVE_CLASS_NUCLEI_TYPES,
        "label_column": "cells_label",
        "ignore_labels": ["less10"],
        "fallback_class": "Background",
    },
    9: {
        "label_mapping": NINE_CLASS_LABEL_MAPPING,
        "nuclei_types": NINE_CLASS_NUCLEI_TYPES,
        "label_column": "cells_label",
        "ignore_labels": ["less10"],
        "fallback_class": "Background",
    },
    2: {
        "label_mapping": CANCER_NORMAL_LABEL_MAPPING,
        "nuclei_types": CANCER_NORMAL_NUCLEI_TYPES,
        "label_column": "cells_label3",
        "ignore_labels": ["less10"],
        "fallback_class": "Background",
    },
}


# ==============================================================================
# Core Processing Functions
# ==============================================================================


def decode_cell_id_map(npz_bytes: bytes) -> np.ndarray:
    """Decode sparse CSR matrix from .npz bytes to dense instance map.

    The HuggingFace dataset stores per-pixel cell instance IDs as compressed
    sparse matrices (CSR format in .npz). Most pixels are background (0),
    making sparse storage very efficient.

    Args:
        npz_bytes: Raw bytes of the .npz file containing a scipy sparse matrix

    Returns:
        Dense (H, W) int32 array where each pixel value is the cell instance ID
    """
    from scipy.sparse import load_npz as sparse_load_npz

    buf = io.BytesIO(npz_bytes)
    sparse_mat = sparse_load_npz(buf)
    return sparse_mat.toarray().astype(np.int32)


def decode_image(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to RGB numpy array.

    Args:
        png_bytes: Raw PNG image bytes

    Returns:
        (H, W, 3) uint8 RGB array
    """
    buf = io.BytesIO(png_bytes)
    img = Image.open(buf).convert("RGB")
    return np.array(img, dtype=np.uint8)


def build_type_map(
    cell_id_map: np.ndarray,
    slide_meta: pd.DataFrame,
    label_column: str,
    label_mapping: Dict[str, str],
    class_to_int: Dict[str, int],
    ignore_labels: List[str],
    fallback_class: str = "Background",
) -> np.ndarray:
    """Convert instance map to per-pixel type map using cell metadata.

    For each unique cell ID in the patch:
    1. Look up the cell's raw label from metadata
    2. Map raw label → target class via label_mapping
    3. Convert class name → integer

    This uses vectorized numpy operations for efficiency.

    Args:
        cell_id_map: (H, W) instance segmentation map
        slide_meta: DataFrame indexed by cell_id_int with label columns
        label_column: Column name in metadata containing cell type labels
        label_mapping: Mapping from raw labels to target class names
        class_to_int: Mapping from class names to integer labels
        ignore_labels: Labels to ignore (treated as background)
        fallback_class: Default class for unmapped/NaN labels

    Returns:
        (H, W) uint8 array with per-pixel class labels
    """
    # Get unique cell IDs and inverse mapping for reconstruction
    ids_all, inv = np.unique(cell_id_map, return_inverse=True)

    # Build class assignment for each unique ID
    class_ints = np.zeros(len(ids_all), dtype=np.uint8)

    for idx, cell_id in enumerate(ids_all):
        if cell_id == 0:
            # Background pixel
            class_ints[idx] = class_to_int["Background"]
            continue

        # Look up this cell's label in metadata
        if cell_id in slide_meta.index:
            raw_label = slide_meta.loc[cell_id, label_column]
        else:
            raw_label = None

        # Handle missing/ignored labels
        if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
            class_ints[idx] = class_to_int[fallback_class]
        elif raw_label in ignore_labels:
            class_ints[idx] = class_to_int["Background"]
        elif raw_label in label_mapping:
            target_class = label_mapping[raw_label]
            class_ints[idx] = class_to_int[target_class]
        else:
            class_ints[idx] = class_to_int[fallback_class]

    # Apply lookup table to reconstruct full type map
    type_map = class_ints[inv].reshape(cell_id_map.shape)
    return type_map


def extract_cell_detections(
    cell_id_map: np.ndarray,
    type_map: np.ndarray,
    class_to_int: Dict[str, int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Extract cell centroids and types from instance + type maps.

    Computes the centroid of each cell instance and assigns the majority
    type label from the type map.

    Args:
        cell_id_map: (H, W) instance segmentation map
        type_map: (H, W) per-pixel type labels
        class_to_int: class name → int mapping (to determine background)

    Returns:
        Tuple of (centroids, types) where:
            centroids: List of (x, y) centroid coordinates
            types: List of integer type labels (0-indexed for classifier)
    """
    bg_int = class_to_int["Background"]
    unique_ids = np.unique(cell_id_map)
    unique_ids = unique_ids[unique_ids != 0]  # Skip background

    centroids = []
    types = []

    for cell_id in unique_ids:
        mask = cell_id_map == cell_id
        ys, xs = np.where(mask)

        if len(ys) == 0:
            continue

        # Compute centroid
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))

        # Get majority type for this cell (excluding background pixels)
        cell_types = type_map[mask]
        cell_types = cell_types[cell_types != bg_int]

        if len(cell_types) == 0:
            continue

        # Use majority vote for the cell type
        unique_types, counts = np.unique(cell_types, return_counts=True)
        majority_type = unique_types[np.argmax(counts)]

        # Convert to 0-indexed for classifier (subtract 1 since Background=0)
        cell_type_idx = int(majority_type) - 1
        if cell_type_idx < 0:
            continue

        centroids.append((cx, cy))
        types.append(cell_type_idx)

    return centroids, types


def assign_splits_spatial(
    patch_info: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    boundary_margin: int = 64,
    axis: str = "x",
) -> pd.DataFrame:
    """Assign train/val/test splits based on spatial coordinates.

    Splits patches along one axis with a margin zone between splits
    to prevent spatial data leakage.

    Args:
        patch_info: DataFrame with 'xmin', 'ymin' columns
        train_frac: Fraction for training
        valid_frac: Fraction for validation
        boundary_margin: Pixels to discard at split boundaries
        axis: Axis to split on ("x" or "y")

    Returns:
        DataFrame with added 'split' column
    """
    coord_col = f"{axis}min" if f"{axis}min" in patch_info.columns else axis

    patch_info = patch_info.copy()
    coords = patch_info[coord_col].values
    cmin, cmax = coords.min(), coords.max()
    span = cmax - cmin

    b1 = cmin + train_frac * span
    b2 = cmin + (train_frac + valid_frac) * span

    splits = []
    for c in coords:
        if c < b1 - boundary_margin:
            splits.append("train")
        elif c > b1 + boundary_margin and c < b2 - boundary_margin:
            splits.append("val")
        elif c > b2 + boundary_margin:
            splits.append("test")
        else:
            splits.append("discard")

    patch_info["split"] = splits
    return patch_info


def assign_splits_slide(
    patch_info: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign splits at the whole-slide level.

    Each slide goes entirely to one split to prevent data leakage.

    Args:
        patch_info: DataFrame with 'slide_id' column
        train_frac: Fraction of slides for training
        valid_frac: Fraction of slides for validation
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with added 'split' column
    """
    rng = np.random.default_rng(random_seed)
    slides = patch_info["slide_id"].unique()
    rng.shuffle(slides)

    n_train = max(1, int(len(slides) * train_frac))
    n_valid = max(1, int(len(slides) * valid_frac))

    train_slides = set(slides[:n_train])
    valid_slides = set(slides[n_train : n_train + n_valid])
    test_slides = set(slides[n_train + n_valid :])

    # If only 2 slides, use spatial split on first, test on second
    if len(slides) == 2:
        train_slides = {slides[0]}
        valid_slides = {slides[0]}  # Will be split spatially later
        test_slides = {slides[1]}

    patch_info = patch_info.copy()
    patch_info["split"] = patch_info["slide_id"].apply(
        lambda s: "train" if s in train_slides
        else "val" if s in valid_slides
        else "test"
    )
    return patch_info


def assign_splits_random(
    patch_info: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign splits randomly (baseline, risk of spatial leakage).

    Args:
        patch_info: DataFrame with patch metadata
        train_frac: Fraction for training
        valid_frac: Fraction for validation
        random_seed: Random seed

    Returns:
        DataFrame with added 'split' column
    """
    rng = np.random.default_rng(random_seed)
    n = len(patch_info)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    splits = np.array(["test"] * n)
    splits[indices[:n_train]] = "train"
    splits[indices[n_train : n_train + n_valid]] = "val"

    patch_info = patch_info.copy()
    patch_info["split"] = splits
    return patch_info


# ==============================================================================
# Main Processing Pipeline
# ==============================================================================


def download_and_process_sthelar(
    output_dir: str,
    magnification: str = "40x",
    num_classes: int = 5,
    split_strategy: str = "spatial",
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    boundary_margin: int = 64,
    max_patches_per_slide: Optional[int] = None,
    tissues: Optional[List[str]] = None,
    slide_ids: Optional[List[str]] = None,
    random_seed: int = 42,
) -> Path:
    """Download STHELAR dataset from HuggingFace and convert to CellViT++ format.

    This function:
    1. Downloads the STHELAR dataset from HuggingFace
    2. Decodes images and cell instance maps from parquet shards
    3. Loads cell metadata to get type labels
    4. Builds per-pixel type maps using the specified label space
    5. Extracts cell centroids and type annotations
    6. Splits data using the specified strategy
    7. Saves in CellViT++ classifier training format

    Args:
        output_dir: Directory to save processed dataset
        magnification: "40x" or "20x"
        num_classes: Number of classes (2, 5, or 9)
        split_strategy: "spatial", "slide", or "random"
        train_frac: Training fraction
        valid_frac: Validation fraction
        boundary_margin: Margin for spatial splits (pixels)
        max_patches_per_slide: Max patches per slide (None = all)
        tissues: Filter to specific tissues (None = all)
        slide_ids: Filter to specific slide IDs (None = all)
        random_seed: Random seed for reproducibility

    Returns:
        Path to the output directory
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get label configuration
    if num_classes not in LABEL_CONFIGS:
        raise ValueError(f"num_classes must be one of {list(LABEL_CONFIGS.keys())}")

    label_config = LABEL_CONFIGS[num_classes]
    label_mapping = label_config["label_mapping"]
    nuclei_types = label_config["nuclei_types"]
    label_column = label_config["label_column"]
    ignore_labels = label_config["ignore_labels"]
    fallback_class = label_config["fallback_class"]

    # Class name → int mapping
    class_to_int = nuclei_types

    # -------------------------------------------------------------------------
    # Step 1: Load dataset from HuggingFace
    # -------------------------------------------------------------------------
    dataset_name = f"FelicieGS/STHELAR_{magnification}"
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info("This may take a while for the first download...")

    ds = load_dataset(dataset_name, split="train")
    logger.info(f"Loaded {len(ds)} patches from HuggingFace")

    # -------------------------------------------------------------------------
    # Step 2: Load patch overview / metadata
    # -------------------------------------------------------------------------
    # Convert to pandas for easier manipulation
    # We need: file_name, slide_id, tissue, xmin, ymin (coordinates)
    logger.info("Extracting patch metadata...")

    patch_info = pd.DataFrame({
        "idx": range(len(ds)),
        "file_name": ds["file_name"] if "file_name" in ds.column_names else [f"patch_{i}" for i in range(len(ds))],
        "slide_id": ds["slide_id"] if "slide_id" in ds.column_names else ["unknown"] * len(ds),
        "tissue": ds["tissue"] if "tissue" in ds.column_names else ["unknown"] * len(ds),
    })

    # Extract coordinates from file_name or use index-based positions
    if "xmin" in ds.column_names:
        patch_info["xmin"] = ds["xmin"]
        patch_info["ymin"] = ds["ymin"]
    else:
        # Try to parse from file_name pattern: {slide}__x{X}_y{Y}__{stem}
        xmins, ymins = [], []
        for fname in patch_info["file_name"]:
            try:
                parts = str(fname).split("__")
                coord_part = [p for p in parts if p.startswith("x") and "_y" in p]
                if coord_part:
                    x_str, y_str = coord_part[0].split("_y")
                    xmins.append(int(x_str[1:]))
                    ymins.append(int(y_str))
                else:
                    xmins.append(0)
                    ymins.append(0)
            except (ValueError, IndexError):
                xmins.append(0)
                ymins.append(0)
        patch_info["xmin"] = xmins
        patch_info["ymin"] = ymins

    # -------------------------------------------------------------------------
    # Step 3: Filter patches
    # -------------------------------------------------------------------------
    if tissues is not None:
        patch_info = patch_info[patch_info["tissue"].isin(tissues)]
        logger.info(f"Filtered to tissues {tissues}: {len(patch_info)} patches")

    if slide_ids is not None:
        patch_info = patch_info[patch_info["slide_id"].isin(slide_ids)]
        logger.info(f"Filtered to slides {slide_ids}: {len(patch_info)} patches")

    if max_patches_per_slide is not None:
        rng = np.random.default_rng(random_seed)
        filtered_dfs = []
        for slide_id, group in patch_info.groupby("slide_id"):
            if len(group) > max_patches_per_slide:
                sampled_idx = rng.choice(len(group), max_patches_per_slide, replace=False)
                filtered_dfs.append(group.iloc[sampled_idx])
            else:
                filtered_dfs.append(group)
        patch_info = pd.concat(filtered_dfs, ignore_index=True)
        logger.info(f"Capped to {max_patches_per_slide}/slide: {len(patch_info)} patches")

    # -------------------------------------------------------------------------
    # Step 4: Assign splits
    # -------------------------------------------------------------------------
    logger.info(f"Assigning splits using '{split_strategy}' strategy")

    if split_strategy == "spatial":
        patch_info = assign_splits_spatial(
            patch_info, train_frac, valid_frac, boundary_margin
        )
    elif split_strategy == "slide":
        patch_info = assign_splits_slide(
            patch_info, train_frac, valid_frac, random_seed
        )
    elif split_strategy == "random":
        patch_info = assign_splits_random(
            patch_info, train_frac, valid_frac, random_seed
        )
    else:
        # Auto: slide if ≥2 slides, spatial otherwise
        n_slides = patch_info["slide_id"].nunique()
        if n_slides >= 2:
            patch_info = assign_splits_slide(
                patch_info, train_frac, valid_frac, random_seed
            )
        else:
            patch_info = assign_splits_spatial(
                patch_info, train_frac, valid_frac, boundary_margin
            )

    # Remove discarded patches (boundary zone)
    patch_info = patch_info[patch_info["split"] != "discard"].reset_index(drop=True)

    split_counts = patch_info["split"].value_counts()
    logger.info(f"Split distribution: {split_counts.to_dict()}")

    # -------------------------------------------------------------------------
    # Step 5: Load cell metadata per slide
    # -------------------------------------------------------------------------
    logger.info("Loading cell metadata...")

    # The HF dataset may include cell_metadata as a separate config or
    # it might need to be downloaded separately
    slide_metadata = {}

    # Try to load from HF dataset features
    if "cell_metadata" in ds.column_names:
        # Metadata embedded in dataset
        for slide_id in patch_info["slide_id"].unique():
            slide_patches = patch_info[patch_info["slide_id"] == slide_id]
            if len(slide_patches) > 0:
                first_idx = slide_patches["idx"].iloc[0]
                meta_bytes = ds[first_idx]["cell_metadata"]
                if meta_bytes is not None:
                    meta_df = pd.read_parquet(io.BytesIO(meta_bytes))
                    meta_df = meta_df.set_index("cell_id_int")
                    slide_metadata[slide_id] = meta_df
    else:
        # Try loading from HF dataset with cell_metadata config
        logger.info("Attempting to load cell metadata from dataset configuration...")
        try:
            meta_ds = load_dataset(dataset_name, "cell_metadata", split="train")
            for row in meta_ds:
                slide_id = row.get("slide_id", "unknown")
                if slide_id not in slide_metadata:
                    slide_metadata[slide_id] = []
                slide_metadata[slide_id].append(row)

            # Convert to DataFrames
            for slide_id in slide_metadata:
                if isinstance(slide_metadata[slide_id], list):
                    df = pd.DataFrame(slide_metadata[slide_id])
                    if "cell_id_int" in df.columns:
                        df = df.set_index("cell_id_int")
                    slide_metadata[slide_id] = df
        except Exception as e:
            logger.warning(
                f"Could not load cell_metadata config: {e}. "
                "Will try to load from local parquet files."
            )
            # Look for local cell_metadata directory
            meta_dir = output_path / "cell_metadata"
            if meta_dir.exists():
                for parquet_file in meta_dir.glob("*.parquet"):
                    slide_id = parquet_file.stem.replace("_cell_metadata", "")
                    meta_df = pd.read_parquet(parquet_file)
                    if "cell_id_int" in meta_df.columns:
                        meta_df = meta_df.set_index("cell_id_int")
                    slide_metadata[slide_id] = meta_df
                    logger.info(f"  Loaded metadata for {slide_id}: {len(meta_df)} cells")

    if not slide_metadata:
        logger.warning(
            "No cell metadata found! Will use instance maps without type labels. "
            "Please download cell_metadata parquet files from the HuggingFace dataset "
            "and place them in: {output_path}/cell_metadata/"
        )

    # -------------------------------------------------------------------------
    # Step 6: Process patches and save
    # -------------------------------------------------------------------------
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "detections").mkdir(parents=True, exist_ok=True)

    # Track cell counts per split
    cell_counts = {"train": [], "val": [], "test": []}

    logger.info(f"Processing {len(patch_info)} patches...")

    for _, row in tqdm(patch_info.iterrows(), total=len(patch_info), desc="Processing"):
        idx = row["idx"]
        split = row["split"]
        slide_id = row["slide_id"]
        file_name = row["file_name"]

        # Create unique patch ID
        patch_uid = f"{slide_id}__{Path(file_name).stem}"

        # Get the raw data from HuggingFace dataset
        sample = ds[idx]

        # Decode image
        if isinstance(sample["image"], Image.Image):
            img = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
        elif isinstance(sample["image"], bytes):
            img = decode_image(sample["image"])
        elif isinstance(sample["image"], dict) and "bytes" in sample["image"]:
            img = decode_image(sample["image"]["bytes"])
        else:
            img = np.array(sample["image"], dtype=np.uint8)

        # Decode cell_id_map (instance segmentation)
        cell_id_map = None
        if "cell_id_map" in sample and sample["cell_id_map"] is not None:
            cell_id_data = sample["cell_id_map"]
            if isinstance(cell_id_data, bytes):
                cell_id_map = decode_cell_id_map(cell_id_data)
            elif isinstance(cell_id_data, dict) and "bytes" in cell_id_data:
                cell_id_map = decode_cell_id_map(cell_id_data["bytes"])
            elif isinstance(cell_id_data, np.ndarray):
                cell_id_map = cell_id_data.astype(np.int32)

        if cell_id_map is None:
            logger.debug(f"Skipping {patch_uid}: no cell_id_map")
            continue

        # Build type map if metadata available
        slide_meta = slide_metadata.get(slide_id)

        if slide_meta is not None and len(slide_meta) > 0:
            type_map = build_type_map(
                cell_id_map=cell_id_map,
                slide_meta=slide_meta,
                label_column=label_column,
                label_mapping=label_mapping,
                class_to_int=class_to_int,
                ignore_labels=ignore_labels,
                fallback_class=fallback_class,
            )
        else:
            # Without metadata, all cells are "Unknown" → use class 1 as fallback
            type_map = np.zeros_like(cell_id_map, dtype=np.uint8)
            type_map[cell_id_map > 0] = 1

        # Extract cell centroids and types
        centroids, types = extract_cell_detections(
            cell_id_map, type_map, class_to_int
        )

        if len(centroids) == 0:
            continue

        # Save image
        img_save_path = output_path / split / "images" / f"{patch_uid}.png"
        Image.fromarray(img).save(img_save_path)

        # Save detections as JSON (matching CoNSeP format)
        # Types are stored 1-indexed in JSON: extract_cell_detections returns 0-indexed,
        # we add 1 here so that STHELARDataset.__getitem__ can subtract 1 back to 0-indexed.
        # This matches the convention used by CoNSeP/Ocelot datasets.
        detections_data = []
        for (cx, cy), ct in zip(centroids, types):
            detections_data.append({
                "centroid": [cx, cy],
                "type": ct + 1,
            })

        det_save_path = output_path / split / "detections" / f"{patch_uid}.json"
        with open(det_save_path, "w") as f:
            json.dump(detections_data, f)

        # Track cell counts
        type_counter = {}
        for ct in types:
            type_counter[ct] = type_counter.get(ct, 0) + 1
        cell_counts[split].append({
            "patch_name": patch_uid,
            "total_cells": len(types),
            **{f"class_{k}": v for k, v in type_counter.items()},
        })

    # -------------------------------------------------------------------------
    # Step 7: Save metadata files
    # -------------------------------------------------------------------------
    # Save cell count CSVs
    for split in ["train", "val", "test"]:
        if cell_counts[split]:
            df = pd.DataFrame(cell_counts[split]).fillna(0)
            df.to_csv(output_path / f"cell_count_{split}.csv", index=False)

    # Save label map
    int_to_class = {v: k for k, v in nuclei_types.items() if k != "Background"}
    label_map = {v - 1: k for k, v in nuclei_types.items() if k != "Background"}

    # Save dataset configuration
    config = {
        "dataset": "STHELAR",
        "magnification": magnification,
        "num_classes": num_classes - 1 if "Background" in nuclei_types else num_classes,
        "nuclei_types": nuclei_types,
        "label_mapping": label_mapping,
        "label_column": label_column,
        "split_strategy": split_strategy,
        "train_frac": train_frac,
        "valid_frac": valid_frac,
        "boundary_margin": boundary_margin,
        "random_seed": random_seed,
        "label_map": label_map,
        "total_patches": len(patch_info),
        "split_distribution": split_counts.to_dict(),
    }

    with open(output_path / "dataset_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save split manifest
    patch_info.to_csv(output_path / "patch_info_with_split.csv", index=False)

    # Generate train/val filelists (CSV with patch names)
    for split in ["train", "val", "test"]:
        split_patches = patch_info[patch_info["split"] == split]
        filelist_path = output_path / f"{split}_filelist.csv"
        with open(filelist_path, "w") as f:
            for _, row in split_patches.iterrows():
                patch_uid = f"{row['slide_id']}__{Path(row['file_name']).stem}"
                f.write(f"{patch_uid}\n")

    logger.info(f"\nDataset saved to: {output_path}")
    logger.info(f"  Train patches: {split_counts.get('train', 0)}")
    logger.info(f"  Val patches: {split_counts.get('val', 0)}")
    logger.info(f"  Test patches: {split_counts.get('test', 0)}")
    logger.info(f"\nTo train a classifier, use the config at:")
    logger.info(f"  docs/sthelar/configs/sthelar_classifier_5class.yaml")

    return output_path


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess STHELAR dataset for CellViT++ training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed dataset",
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default="40x",
        choices=["20x", "40x"],
        help="Magnification level (default: 40x)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        choices=[2, 5, 9],
        help="Number of classes: 2 (cancer/normal), 5 (grouped), 9 (fine-grained)",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="spatial",
        choices=["spatial", "slide", "random", "auto"],
        help="Split strategy (default: spatial)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="Training fraction (default: 0.7)",
    )
    parser.add_argument(
        "--valid_frac",
        type=float,
        default=0.15,
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--boundary_margin",
        type=int,
        default=64,
        help="Boundary margin for spatial splits in pixels (default: 64)",
    )
    parser.add_argument(
        "--max_patches_per_slide",
        type=int,
        default=None,
        help="Maximum patches per slide (default: all)",
    )
    parser.add_argument(
        "--tissues",
        nargs="+",
        default=None,
        help="Filter to specific tissues (e.g., breast lung skin)",
    )
    parser.add_argument(
        "--slide_ids",
        nargs="+",
        default=None,
        help="Filter to specific slide IDs (e.g., breast_s0 lung_s1)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    download_and_process_sthelar(
        output_dir=args.output_dir,
        magnification=args.magnification,
        num_classes=args.num_classes,
        split_strategy=args.split_strategy,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        boundary_margin=args.boundary_margin,
        max_patches_per_slide=args.max_patches_per_slide,
        tissues=args.tissues,
        slide_ids=args.slide_ids,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
