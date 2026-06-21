# -*- coding: utf-8 -*-
# STHELAR Dataset Statistics Script
#
# Computes statistics of cell annotations across different class schemes
# (2-class, 5-class, 9-class) from the STHELAR dataset.
#
# Usage:
#   python -m cellvit.training.datasets.sthelar.sthelar_statistics \
#       --output_file /path/to/statistics.csv \
#       --magnification 40x
#
# Output:
#   A CSV file with columns: 2-class, 5-class, 9-class, number_of_cells

import argparse
import io
import logging
from pathlib import Path

import pandas as pd

from cellvit.training.datasets.sthelar.prepare_sthelar import (
    CANCER_NORMAL_LABEL_MAPPING,
    FIVE_CLASS_LABEL_MAPPING,
    NINE_CLASS_LABEL_MAPPING,
    LABEL_CONFIGS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def compute_statistics(
    magnification: str = "40x",
    output_file: str = "sthelar_statistics.csv",
) -> pd.DataFrame:
    """Compute cell annotation statistics across 2-class, 5-class, and 9-class schemes.

    Downloads the STHELAR dataset cell metadata from HuggingFace, then counts cells
    for each combination of (2-class, 5-class, 9-class) labels.

    Args:
        magnification: "40x" or "20x"
        output_file: Path to save the output CSV

    Returns:
        DataFrame with columns: 2-class, 5-class, 9-class, number_of_cells
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        )

    dataset_name = f"FelicieGS/STHELAR_{magnification}"
    logger.info(f"Loading dataset: {dataset_name}")

    # Load cell metadata
    label_column_9class = LABEL_CONFIGS[9]["label_column"]  # "cells_label"
    label_column_2class = LABEL_CONFIGS[2]["label_column"]  # "cells_label3"
    ignore_labels = LABEL_CONFIGS[9]["ignore_labels"]

    # Try to load cell metadata from the dataset
    cell_records = []

    try:
        meta_ds = load_dataset(dataset_name, "cell_metadata", split="train")
        logger.info(f"Loaded cell metadata: {len(meta_ds)} records")

        for row in meta_ds:
            raw_9class = row.get(label_column_9class)
            raw_2class = row.get(label_column_2class)

            # Skip ignored labels
            if raw_9class in ignore_labels or raw_2class in ignore_labels:
                continue
            if raw_9class is None or raw_2class is None:
                continue

            # Map 9-class → 5-class
            five_class = FIVE_CLASS_LABEL_MAPPING.get(raw_9class)
            # Map raw 2-class label
            two_class = CANCER_NORMAL_LABEL_MAPPING.get(raw_2class)

            if five_class is None or two_class is None:
                continue

            cell_records.append({
                "9-class": raw_9class,
                "5-class": five_class,
                "2-class": two_class,
            })
    except Exception as e:
        logger.warning(f"Could not load cell_metadata config: {e}")
        logger.info("Attempting to load from main dataset split...")

        # Fallback: load from main split if metadata is embedded
        ds = load_dataset(dataset_name, split="train")

        if "cell_metadata" in ds.column_names:
            for sample in ds:
                meta_bytes = sample.get("cell_metadata")
                if meta_bytes is None:
                    continue
                if isinstance(meta_bytes, dict) and "bytes" in meta_bytes:
                    meta_bytes = meta_bytes["bytes"]
                if not isinstance(meta_bytes, bytes):
                    continue

                meta_df = pd.read_parquet(io.BytesIO(meta_bytes))

                for _, cell_row in meta_df.iterrows():
                    raw_9class = cell_row.get(label_column_9class)
                    raw_2class = cell_row.get(label_column_2class)

                    if raw_9class in ignore_labels or raw_2class in ignore_labels:
                        continue
                    if pd.isna(raw_9class) or pd.isna(raw_2class):
                        continue

                    five_class = FIVE_CLASS_LABEL_MAPPING.get(raw_9class)
                    two_class = CANCER_NORMAL_LABEL_MAPPING.get(raw_2class)

                    if five_class is None or two_class is None:
                        continue

                    cell_records.append({
                        "9-class": raw_9class,
                        "5-class": five_class,
                        "2-class": two_class,
                    })
        else:
            raise RuntimeError(
                "Could not find cell metadata in the dataset. "
                "Ensure the dataset has a 'cell_metadata' config or embedded metadata."
            )

    if not cell_records:
        raise RuntimeError("No valid cell records found in the dataset.")

    # Aggregate statistics
    df = pd.DataFrame(cell_records)
    stats = (
        df.groupby(["2-class", "5-class", "9-class"])
        .size()
        .reset_index(name="number_of_cells")
    )
    stats = stats.sort_values(
        ["2-class", "5-class", "9-class"]
    ).reset_index(drop=True)

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False)

    logger.info(f"Statistics saved to: {output_path}")
    logger.info(f"Total cells: {stats['number_of_cells'].sum()}")
    logger.info(f"\n{stats.to_string(index=False)}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute STHELAR cell annotation statistics across class schemes"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="sthelar_statistics.csv",
        help="Path to save the output CSV (default: sthelar_statistics.csv)",
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default="40x",
        choices=["20x", "40x"],
        help="Magnification level (default: 40x)",
    )

    args = parser.parse_args()

    compute_statistics(
        magnification=args.magnification,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
