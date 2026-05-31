# -*- coding: utf-8 -*-
# ORION-CRC Dataset Preprocessing Script
#
# Downloads and processes the ORION-CRC dataset (from Zenodo) to the format
# expected by CellViT++ classifier training.
#
# Dataset sources:
#   - Processed tiles: https://zenodo.org/records/15340874
#     (ORIONCRC_dataset_tile_20x.zip, ~127 GB)
#   - Original WSIs: https://zenodo.org/records/7637988
#
# Reference:
#   Lin J. et al., "Multiplexed 3D atlas of state transitions and immune
#   interactions in colorectal cancer", Nature Cancer (2023)
#   doi: 10.1038/s43018-023-00576-1
#
# Usage:
#   python -m cellvit.training.datasets.orion_crc.prepare_orion_crc \
#       --input_dir /path/to/ORIONCRC_dataset_tile_20x \
#       --output_dir /path/to/output \
#       --num_classes 7
#
# The input_dir should contain the extracted ORIONCRC_dataset_tile_20x.zip:
#   ORIONCRC_dataset_tile_20x/
#   ├── he/                    (H&E tile images as JPEG)
#   ├── nuclei/                (CellPose instance masks as TIFF)
#   ├── nuclei_csv/            (Per-slide single-cell CSV files)
#   ├── train_dataframe.csv
#   ├── val_dataframe.csv
#   ├── test_dataframe.csv
#   └── slide_dataframe.csv
#
# Output structure (for classifier training):
#   output_dir/
#   ├── train/
#   │   ├── images/       (*.png, 512x512 RGB)
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
import json
import logging
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

# 15 binary marker positivity columns in nuclei_csv files
MARKER_COLUMNS = [
    "CD31_pos",
    "CD45_pos",
    "CD68_pos",
    "CD4_pos",
    "FOXP3_pos",
    "CD8a_pos",
    "CD45RO_pos",
    "CD20_pos",
    "PD-L1_pos",
    "CD3e_pos",
    "CD163_pos",
    "E-cadherin_pos",
    "Ki67_pos",
    "Pan-CK_pos",
    "SMA_pos",
]

# 7-class scheme: biologically meaningful cell types for CRC
# Derived from marker positivity using hierarchical gating logic
SEVEN_CLASS_NUCLEI_TYPES = {
    "Background": 0,
    "Tumor": 1,
    "T_cell": 2,
    "Macrophage": 3,
    "B_cell": 4,
    "Stromal": 5,
    "Endothelial": 6,
    "Other": 7,
}

# 5-class scheme: broader grouping
FIVE_CLASS_NUCLEI_TYPES = {
    "Background": 0,
    "Tumor": 1,
    "Immune": 2,
    "Stromal": 3,
    "Other": 4,
}

# 3-class scheme: tumor vs stroma vs immune
THREE_CLASS_NUCLEI_TYPES = {
    "Background": 0,
    "Tumor": 1,
    "Immune": 2,
    "Stromal": 3,
}

# Mapping from 7-class to 5-class
SEVEN_TO_FIVE_MAPPING = {
    "Tumor": "Tumor",
    "T_cell": "Immune",
    "Macrophage": "Immune",
    "B_cell": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Other",
    "Other": "Other",
}

# Mapping from 7-class to 3-class
SEVEN_TO_THREE_MAPPING = {
    "Tumor": "Tumor",
    "T_cell": "Immune",
    "Macrophage": "Immune",
    "B_cell": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Other": "Stromal",
}

LABEL_CONFIGS = {
    7: {
        "nuclei_types": SEVEN_CLASS_NUCLEI_TYPES,
        "mapping_from_7class": None,  # Identity
    },
    5: {
        "nuclei_types": FIVE_CLASS_NUCLEI_TYPES,
        "mapping_from_7class": SEVEN_TO_FIVE_MAPPING,
    },
    3: {
        "nuclei_types": THREE_CLASS_NUCLEI_TYPES,
        "mapping_from_7class": SEVEN_TO_THREE_MAPPING,
    },
}


# ==============================================================================
# Cell Type Classification from Marker Positivity
# ==============================================================================


def classify_cell_7class(row: pd.Series) -> str:
    """Classify a single cell into 7-class scheme using marker positivity.

    Uses hierarchical gating logic based on biological marker relationships:
    - Tumor/Epithelial: Pan-CK+ or E-cadherin+, and CD45-
    - T cells: CD3e+ (includes CD4+, CD8a+, FOXP3+ subsets)
    - Macrophages: CD68+ or CD163+
    - B cells: CD20+
    - Stromal: SMA+, CD45-
    - Endothelial: CD31+
    - Other: none of the above

    Args:
        row: pandas Series with binary marker positivity columns

    Returns:
        Cell type string (one of 7 classes)
    """
    # Epithelial/Tumor: Pan-CK+ or E-cadherin+, NOT immune (CD45-)
    is_epithelial = bool(row.get("Pan-CK_pos", 0)) or bool(row.get("E-cadherin_pos", 0))
    is_immune = bool(row.get("CD45_pos", 0))

    if is_epithelial and not is_immune:
        return "Tumor"

    # Immune cells (CD45+ or specific immune markers)
    # T cells: CD3e+
    if bool(row.get("CD3e_pos", 0)):
        return "T_cell"

    # Macrophages: CD68+ or CD163+
    if bool(row.get("CD68_pos", 0)) or bool(row.get("CD163_pos", 0)):
        return "Macrophage"

    # B cells: CD20+
    if bool(row.get("CD20_pos", 0)):
        return "B_cell"

    # Other immune (CD45+ but no specific marker)
    if is_immune:
        return "Other"

    # Endothelial: CD31+
    if bool(row.get("CD31_pos", 0)):
        return "Endothelial"

    # Stromal: SMA+
    if bool(row.get("SMA_pos", 0)):
        return "Stromal"

    return "Other"


def classify_cells_vectorized(cell_df: pd.DataFrame) -> pd.Series:
    """Vectorized cell classification for efficiency.

    Args:
        cell_df: DataFrame with binary marker positivity columns

    Returns:
        Series of cell type strings
    """
    # Initialize all as "Other"
    cell_types = pd.Series("Other", index=cell_df.index)

    # Helper for safe column access
    def get_col(name):
        if name in cell_df.columns:
            return cell_df[name].astype(bool)
        return pd.Series(False, index=cell_df.index)

    pan_ck = get_col("Pan-CK_pos")
    e_cad = get_col("E-cadherin_pos")
    cd45 = get_col("CD45_pos")
    cd3e = get_col("CD3e_pos")
    cd68 = get_col("CD68_pos")
    cd163 = get_col("CD163_pos")
    cd20 = get_col("CD20_pos")
    cd31 = get_col("CD31_pos")
    sma = get_col("SMA_pos")

    # Apply hierarchical logic (order matters - later assignments override earlier)
    # Stromal: SMA+, CD45-
    cell_types[(sma) & (~cd45)] = "Stromal"

    # Endothelial: CD31+
    cell_types[cd31] = "Endothelial"

    # B cells: CD20+
    cell_types[cd20] = "B_cell"

    # Macrophages: CD68+ or CD163+
    cell_types[(cd68) | (cd163)] = "Macrophage"

    # T cells: CD3e+
    cell_types[cd3e] = "T_cell"

    # Tumor/Epithelial: (Pan-CK+ or E-cadherin+) and NOT CD45+
    cell_types[((pan_ck) | (e_cad)) & (~cd45)] = "Tumor"

    return cell_types


# ==============================================================================
# Core Processing Functions
# ==============================================================================


def load_nuclei_csv(nuclei_csv_path: Path) -> Dict[str, pd.DataFrame]:
    """Load per-slide nuclei CSV files.

    Args:
        nuclei_csv_path: Path to the nuclei_csv/ directory

    Returns:
        Dictionary mapping slide name to cell DataFrame
    """
    slide_cells = {}
    csv_files = sorted(nuclei_csv_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {nuclei_csv_path}. "
            "Ensure the ORION-CRC dataset is properly extracted."
        )

    logger.info(f"Loading {len(csv_files)} nuclei CSV files...")
    for csv_file in tqdm(csv_files, desc="Loading nuclei CSVs"):
        slide_name = csv_file.stem
        df = pd.read_csv(csv_file)
        slide_cells[slide_name] = df

    return slide_cells


def parse_tile_name(tile_path: str) -> Tuple[str, int, int]:
    """Parse tile filename to extract slide name and coordinates.

    Tile names follow the pattern: {slide_name}_{x}_{y}_{level}_{w}_{h}.jpeg

    Args:
        tile_path: Path string or filename of the tile

    Returns:
        Tuple of (slide_name, x_coord, y_coord)
    """
    fname = Path(tile_path).stem
    parts = fname.rsplit("_", 5)

    if len(parts) >= 6:
        slide_name = parts[0]
        x = int(parts[1])
        y = int(parts[2])
    else:
        # Fallback: try splitting differently
        # Some names may have underscores in slide_name
        # Pattern: everything before last 5 underscore-separated numbers
        all_parts = fname.split("_")
        # Last 5 parts are: x, y, level, w, h
        if len(all_parts) > 5:
            slide_name = "_".join(all_parts[:-5])
            x = int(all_parts[-5])
            y = int(all_parts[-4])
        else:
            slide_name = fname
            x, y = 0, 0

    return slide_name, x, y


def get_cells_in_tile(
    slide_cells: pd.DataFrame,
    tile_x: int,
    tile_y: int,
    tile_size: int = 512,
) -> pd.DataFrame:
    """Get cells that fall within a specific tile's coordinates.

    Args:
        slide_cells: DataFrame with 'x', 'y' columns for cell positions
        tile_x: Top-left x coordinate of the tile in WSI space
        tile_y: Top-left y coordinate of the tile in WSI space
        tile_size: Size of the tile in pixels

    Returns:
        DataFrame of cells within the tile, with local coordinates
    """
    mask = (
        (slide_cells["x"] >= tile_x)
        & (slide_cells["x"] < tile_x + tile_size)
        & (slide_cells["y"] >= tile_y)
        & (slide_cells["y"] < tile_y + tile_size)
    )
    cells_in_tile = slide_cells[mask].copy()

    # Convert to tile-local coordinates
    cells_in_tile["local_x"] = cells_in_tile["x"] - tile_x
    cells_in_tile["local_y"] = cells_in_tile["y"] - tile_y

    return cells_in_tile


# ==============================================================================
# Main Processing Pipeline
# ==============================================================================


def process_orion_crc(
    input_dir: str,
    output_dir: str,
    num_classes: int = 7,
    tile_size: int = 512,
    min_cells_per_tile: int = 5,
    max_tiles_per_split: Optional[int] = None,
    random_seed: int = 42,
) -> Path:
    """Process ORION-CRC dataset into CellViT++ classifier training format.

    This function:
    1. Reads the pre-split tile dataframes (train/val/test)
    2. Loads per-slide nuclei CSV files with marker positivity
    3. Classifies cells into the specified class scheme
    4. Extracts cell centroids within each H&E tile
    5. Saves images and detection annotations in CellViT++ format

    Args:
        input_dir: Path to extracted ORIONCRC_dataset_tile_20x directory
        output_dir: Directory to save processed dataset
        num_classes: Number of classes (3, 5, or 7)
        tile_size: Tile size in pixels (default: 512)
        min_cells_per_tile: Minimum cells required per tile (skip otherwise)
        max_tiles_per_split: Maximum tiles per split (None = all)
        random_seed: Random seed for subsampling

    Returns:
        Path to the output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate input directory
    he_dir = input_path / "he"
    nuclei_csv_dir = input_path / "nuclei_csv"

    if not he_dir.exists():
        raise FileNotFoundError(
            f"H&E tile directory not found: {he_dir}. "
            "Please extract ORIONCRC_dataset_tile_20x.zip first."
        )
    if not nuclei_csv_dir.exists():
        raise FileNotFoundError(
            f"Nuclei CSV directory not found: {nuclei_csv_dir}. "
            "Please extract ORIONCRC_dataset_tile_20x.zip first."
        )

    # Get label configuration
    if num_classes not in LABEL_CONFIGS:
        raise ValueError(f"num_classes must be one of {list(LABEL_CONFIGS.keys())}")

    label_config = LABEL_CONFIGS[num_classes]
    nuclei_types = label_config["nuclei_types"]
    class_mapping = label_config["mapping_from_7class"]

    # -------------------------------------------------------------------------
    # Step 1: Load split dataframes
    # -------------------------------------------------------------------------
    logger.info("Loading split dataframes...")

    split_dfs = {}
    for split in ["train", "val", "test"]:
        split_csv = input_path / f"{split}_dataframe.csv"
        if split_csv.exists():
            split_dfs[split] = pd.read_csv(split_csv)
            logger.info(f"  {split}: {len(split_dfs[split])} tiles")
        else:
            logger.warning(f"  {split}_dataframe.csv not found, skipping split")

    if not split_dfs:
        raise FileNotFoundError(
            "No split dataframes found. Expected train_dataframe.csv, "
            "val_dataframe.csv, and/or test_dataframe.csv"
        )

    # -------------------------------------------------------------------------
    # Step 2: Load nuclei cell data
    # -------------------------------------------------------------------------
    slide_cells = load_nuclei_csv(nuclei_csv_dir)
    logger.info(f"Loaded cell data for {len(slide_cells)} slides")

    # Classify all cells using the 7-class scheme
    logger.info("Classifying cells from marker positivity...")
    for slide_name, cell_df in tqdm(slide_cells.items(), desc="Classifying"):
        cell_df["cell_type_7class"] = classify_cells_vectorized(cell_df)
        slide_cells[slide_name] = cell_df

    # -------------------------------------------------------------------------
    # Step 3: Process tiles per split
    # -------------------------------------------------------------------------
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "detections").mkdir(parents=True, exist_ok=True)

    cell_counts = {"train": [], "val": [], "test": []}
    split_stats = {}
    rng = np.random.default_rng(random_seed)

    for split, split_df in split_dfs.items():
        logger.info(f"\nProcessing {split} split ({len(split_df)} tiles)...")

        # Subsample if requested
        if max_tiles_per_split is not None and len(split_df) > max_tiles_per_split:
            indices = rng.choice(len(split_df), max_tiles_per_split, replace=False)
            split_df = split_df.iloc[indices].reset_index(drop=True)
            logger.info(f"  Subsampled to {len(split_df)} tiles")

        processed_count = 0
        skipped_count = 0

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{split}"):
            # Get tile path - try common column names
            tile_path = None
            for col in ["image_path", "he_path", "in_path"]:
                if col in row and pd.notna(row[col]):
                    tile_path = row[col]
                    break

            if tile_path is None:
                # Try to reconstruct from slide name and coordinates
                slide_name = row.get("in_slide_name", row.get("slide_name", ""))
                if not slide_name:
                    skipped_count += 1
                    continue
                # Look for matching H&E tile
                tile_candidates = list(he_dir.glob(f"{slide_name}*.jpeg")) + \
                                  list(he_dir.glob(f"{slide_name}*.jpg")) + \
                                  list(he_dir.glob(f"{slide_name}*.png"))
                if not tile_candidates:
                    skipped_count += 1
                    continue
                tile_path = str(tile_candidates[0].name)

            # Resolve full path to H&E tile
            tile_filename = Path(tile_path).name
            full_tile_path = he_dir / tile_filename
            if not full_tile_path.exists():
                # Try relative path from input dir
                full_tile_path = input_path / tile_path
                if not full_tile_path.exists():
                    skipped_count += 1
                    continue

            # Parse tile coordinates
            slide_name, tile_x, tile_y = parse_tile_name(tile_filename)

            # Find slide in nuclei data
            # Try exact match first, then prefix match
            cell_df = None
            if slide_name in slide_cells:
                cell_df = slide_cells[slide_name]
            else:
                # Try to find matching slide by prefix
                for sname in slide_cells:
                    if slide_name.startswith(sname) or sname.startswith(slide_name):
                        cell_df = slide_cells[sname]
                        break

            if cell_df is None or len(cell_df) == 0:
                skipped_count += 1
                continue

            # Get cells in this tile
            cells_in_tile = get_cells_in_tile(cell_df, tile_x, tile_y, tile_size)

            if len(cells_in_tile) < min_cells_per_tile:
                skipped_count += 1
                continue

            # Map cell types to target class scheme
            if class_mapping is not None:
                cell_type_labels = cells_in_tile["cell_type_7class"].map(class_mapping)
            else:
                cell_type_labels = cells_in_tile["cell_type_7class"]

            # Convert to integer labels
            class_to_int = nuclei_types
            cell_type_ints = cell_type_labels.map(class_to_int)

            # Filter out cells with unmapped types (NaN) or Background
            valid_mask = cell_type_ints.notna() & (cell_type_ints > 0)
            valid_cells = cells_in_tile[valid_mask]
            valid_types = cell_type_ints[valid_mask].astype(int)

            if len(valid_cells) < min_cells_per_tile:
                skipped_count += 1
                continue

            # Create unique tile ID
            tile_uid = f"{slide_name}_x{tile_x}_y{tile_y}"

            # Save image as PNG
            try:
                img = Image.open(full_tile_path).convert("RGB")
            except Exception as e:
                logger.debug(f"Failed to open {full_tile_path}: {e}")
                skipped_count += 1
                continue

            img_save_path = output_path / split / "images" / f"{tile_uid}.png"
            img.save(img_save_path)

            # Save detections as JSON (1-indexed types, matching CellViT++ convention)
            detections_data = []
            for idx, cell_row in valid_cells.iterrows():
                cx = int(cell_row["local_x"])
                cy = int(cell_row["local_y"])
                ct = int(valid_types.loc[idx])
                detections_data.append({
                    "centroid": [cx, cy],
                    "type": ct,  # Already 1-indexed from nuclei_types dict
                })

            det_save_path = output_path / split / "detections" / f"{tile_uid}.json"
            with open(det_save_path, "w") as f:
                json.dump(detections_data, f)

            # Track cell counts
            type_counter = {}
            for ct in valid_types:
                type_counter[ct] = type_counter.get(ct, 0) + 1
            cell_counts[split].append({
                "patch_name": tile_uid,
                "total_cells": len(valid_cells),
                **{f"class_{k}": v for k, v in type_counter.items()},
            })

            processed_count += 1

        split_stats[split] = {
            "processed": processed_count,
            "skipped": skipped_count,
        }
        logger.info(
            f"  {split}: processed {processed_count} tiles, skipped {skipped_count}"
        )

    # -------------------------------------------------------------------------
    # Step 4: Save metadata files
    # -------------------------------------------------------------------------
    # Save cell count CSVs
    for split in ["train", "val", "test"]:
        if cell_counts[split]:
            df = pd.DataFrame(cell_counts[split]).fillna(0)
            df.to_csv(output_path / f"cell_count_{split}.csv", index=False)

    # Build label map (0-indexed for classifier)
    label_map = {v - 1: k for k, v in nuclei_types.items() if k != "Background"}

    # Save dataset configuration
    config = {
        "dataset": "ORION-CRC",
        "source": "https://zenodo.org/records/15340874",
        "magnification": "20x",
        "resolution_mpp": 0.5,
        "tile_size": tile_size,
        "num_classes": len(nuclei_types) - 1,  # Exclude Background
        "nuclei_types": nuclei_types,
        "label_map": label_map,
        "min_cells_per_tile": min_cells_per_tile,
        "random_seed": random_seed,
        "split_statistics": split_stats,
    }

    with open(output_path / "dataset_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save file lists
    for split in ["train", "val", "test"]:
        if cell_counts[split]:
            filelist_path = output_path / f"{split}_filelist.csv"
            with open(filelist_path, "w") as f:
                for entry in cell_counts[split]:
                    f.write(f"{entry['patch_name']}\n")

    logger.info(f"\nDataset saved to: {output_path}")
    for split, stats in split_stats.items():
        logger.info(f"  {split}: {stats['processed']} tiles")
    logger.info(f"  Label scheme: {num_classes}-class")
    logger.info(f"  Classes: {list(nuclei_types.keys())}")

    return output_path


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Process ORION-CRC dataset for CellViT++ classifier training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to extracted ORIONCRC_dataset_tile_20x directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed dataset",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=7,
        choices=[3, 5, 7],
        help="Number of classes: 7 (fine-grained), 5 (grouped), 3 (tumor/immune/stromal)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512)",
    )
    parser.add_argument(
        "--min_cells_per_tile",
        type=int,
        default=5,
        help="Minimum cells per tile to include (default: 5)",
    )
    parser.add_argument(
        "--max_tiles_per_split",
        type=int,
        default=None,
        help="Maximum tiles per split for subsampling (default: all)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    process_orion_crc(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        tile_size=args.tile_size,
        min_cells_per_tile=args.min_cells_per_tile,
        max_tiles_per_split=args.max_tiles_per_split,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
