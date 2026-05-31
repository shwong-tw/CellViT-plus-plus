# -*- coding: utf-8 -*-
# STHELAR Dataset Relabeling Script
#
# Relabels the STHELAR dataset to a custom class scheme based on a user-provided
# CSV mapping from 9-class labels to new class labels.
#
# Usage:
#   python -m cellvit.training.datasets.sthelar.sthelar_relabel \
#       --input_dir /path/to/processed_sthelar \
#       --mapping_file /path/to/mapping.csv \
#       --output_dir /path/to/relabeled_output
#
# The mapping CSV must have columns: 9-class, new-class
# Example:
#   9-class,new-class
#   Epithelial,Epithelial
#   Blood_vessel,Stromal
#   Fibroblast_Myofibroblast,Stromal
#   Myeloid,Immune
#   B_Plasma,Immune
#   T_NK,Immune
#   Melanocyte,Other
#   Specialized,Other
#   Other,Other

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from cellvit.training.datasets.sthelar.prepare_sthelar import (
    NINE_CLASS_NUCLEI_TYPES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_mapping(mapping_file: str) -> Dict[str, str]:
    """Load the relabeling mapping from a CSV file.

    Args:
        mapping_file: Path to CSV with columns '9-class' and 'new-class'

    Returns:
        Dictionary mapping 9-class label names to new class names
    """
    df = pd.read_csv(mapping_file)

    if "9-class" not in df.columns or "new-class" not in df.columns:
        raise ValueError(
            f"Mapping file must have columns '9-class' and 'new-class'. "
            f"Found columns: {list(df.columns)}"
        )

    mapping = dict(zip(df["9-class"], df["new-class"]))

    # Validate that all 9-class labels are covered
    expected_labels = set(NINE_CLASS_NUCLEI_TYPES.keys()) - {"Background"}
    provided_labels = set(mapping.keys())
    missing = expected_labels - provided_labels

    if missing:
        logger.warning(
            f"Mapping is missing entries for: {missing}. "
            "These will be mapped to 'Background' (excluded)."
        )

    return mapping


def build_new_class_config(mapping: Dict[str, str]) -> Dict[str, int]:
    """Build the new nuclei_types dictionary from the mapping.

    Args:
        mapping: 9-class → new-class mapping

    Returns:
        Dictionary mapping new class names to integer IDs (Background=0, then 1, 2, ...)
    """
    new_classes = sorted(set(mapping.values()))
    nuclei_types = {"Background": 0}
    for i, cls in enumerate(new_classes, start=1):
        nuclei_types[cls] = i
    return nuclei_types


def relabel_dataset(
    input_dir: str,
    mapping_file: str,
    output_dir: str,
) -> Path:
    """Relabel a processed STHELAR dataset to a new class scheme.

    Reads the existing processed dataset (with detection JSONs containing
    cell types in 9-class scheme), applies the mapping to convert to the
    new class scheme, and saves a new dataset.

    Args:
        input_dir: Path to the processed STHELAR dataset directory
        mapping_file: Path to CSV mapping file (columns: 9-class, new-class)
        output_dir: Path to save the relabeled dataset

    Returns:
        Path to the output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Load the existing dataset config
    config_path = input_path / "dataset_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            existing_config = yaml.safe_load(f)
    else:
        existing_config = {}

    # Verify the input is a 9-class dataset or determine the source scheme
    existing_nuclei_types = existing_config.get("nuclei_types", NINE_CLASS_NUCLEI_TYPES)

    # Build old class int → old class name mapping (excluding Background)
    old_int_to_name = {v: k for k, v in existing_nuclei_types.items() if k != "Background"}

    # Load and validate the mapping
    mapping = load_mapping(mapping_file)
    logger.info(f"Loaded mapping: {mapping}")

    # Build new class configuration
    new_nuclei_types = build_new_class_config(mapping)
    logger.info(f"New class scheme: {new_nuclei_types}")

    # Build conversion: old_type_int (1-indexed in JSON) → new_type_int (1-indexed in JSON)
    # In detection JSONs, type is stored as 1-indexed (class_int from nuclei_types)
    # old JSON type = nuclei_types[old_class_name] (already 1-indexed since Background=0)
    old_to_new_type = {}
    for old_name, old_int in existing_nuclei_types.items():
        if old_name == "Background":
            continue
        if old_name in mapping:
            new_name = mapping[old_name]
            new_int = new_nuclei_types[new_name]
            old_to_new_type[old_int] = new_int
        else:
            # Not in mapping → exclude (map to 0 / Background)
            old_to_new_type[old_int] = 0

    logger.info(f"Type conversion (old → new): {old_to_new_type}")

    # Process each split
    output_path.mkdir(parents=True, exist_ok=True)
    total_cells = 0
    excluded_cells = 0

    for split in ["train", "val", "test"]:
        split_input = input_path / split
        if not split_input.exists():
            continue

        # Create output directories
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "detections").mkdir(parents=True, exist_ok=True)

        # Copy images (unchanged)
        images_dir = split_input / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.png"):
                shutil.copy2(img_file, output_path / split / "images" / img_file.name)

        # Relabel detections
        detections_dir = split_input / "detections"
        if not detections_dir.exists():
            continue

        for det_file in detections_dir.glob("*.json"):
            with open(det_file) as f:
                detections = json.load(f)

            new_detections = []
            for det in detections:
                old_type = det["type"]
                new_type = old_to_new_type.get(old_type, 0)
                total_cells += 1

                if new_type == 0:
                    # Mapped to Background → exclude
                    excluded_cells += 1
                    continue

                new_detections.append({
                    "centroid": det["centroid"],
                    "type": new_type,
                })

            # Save relabeled detections
            det_output = output_path / split / "detections" / det_file.name
            with open(det_output, "w") as f:
                json.dump(new_detections, f)

    # Save new dataset config
    new_config = {
        "dataset": "STHELAR",
        "magnification": existing_config.get("magnification", "40x"),
        "num_classes": len(new_nuclei_types) - 1,  # Exclude Background
        "nuclei_types": new_nuclei_types,
        "label_mapping": mapping,
        "source_scheme": "9-class",
        "split_strategy": existing_config.get("split_strategy", "unknown"),
        "label_map": {v - 1: k for k, v in new_nuclei_types.items() if k != "Background"},
        "total_cells_processed": total_cells,
        "cells_excluded": excluded_cells,
    }

    with open(output_path / "dataset_config.yaml", "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

    # Copy over other metadata files if they exist
    for meta_file in ["patch_info_with_split.csv", "train_filelist.csv",
                      "val_filelist.csv", "test_filelist.csv"]:
        src = input_path / meta_file
        if src.exists():
            shutil.copy2(src, output_path / meta_file)

    logger.info(f"\nRelabeled dataset saved to: {output_path}")
    logger.info(f"  Total cells processed: {total_cells}")
    logger.info(f"  Cells excluded (mapped to Background): {excluded_cells}")
    logger.info(f"  Cells retained: {total_cells - excluded_cells}")
    logger.info(f"  New class scheme: {new_nuclei_types}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Relabel STHELAR dataset to a custom class scheme"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the processed STHELAR dataset (9-class)",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        required=True,
        help="Path to CSV mapping file with columns: 9-class, new-class",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the relabeled dataset",
    )

    args = parser.parse_args()

    relabel_dataset(
        input_dir=args.input_dir,
        mapping_file=args.mapping_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
