# -*- coding: utf-8 -*-
# CRC CODEX Inference Code
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

"""
CRC CODEX-specific evaluation script.

⚠️  IMPORTANT: This script is designed specifically for CRC CODEX datasets
    with .npy format containing inst_map and type_map dictionaries.
    
    Dataset structure:
        - Files: {images,labels}/*.npy
        - Image size: 256x256 pixel patches
        - Labels loaded with: np.load(label, allow_pickle=True).item()
        - Contains: inst_map and type_map
        - 3 nuclei types: 1 (Connective), 2 (Inflammatory), 3 (Neoplastic)

This script evaluates models on datasets with:
    - CRC CODEX-specific .npy format
    - 256x256 pixel patches
    - Nuclei type labels (no tissue types)
    - Instance segmentation masks
    - Standard segmentation metrics (Dice, AJI, PQ)
"""


import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)

import argparse
import json
from pathlib import Path
from typing import Callable, List, Tuple, Union

import cv2
import numpy as np
import pycm
import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.crc_codex import CRCCodexDataset
from cellvit.training.utils.metrics import (
    binarize,
    cell_detection_scores,
    cell_type_detection_scores,
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_pq,
    get_pq,
    remap_label,
)
from cellvit.training.utils.post_proc_cellvit import calculate_instances
from cellvit.training.utils.tools import pair_coordinates
from scipy.io import loadmat
from PIL import Image


class CellViTInfExpCRCCodex(CellViTClassifierInferenceExperiment):
    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load CRC CODEX Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: CRC CODEX Dataset
        """
        dataset = CRCCodexDataset(
            dataset_path=self.dataset_path,
            split=self.split,  # Use configurable split name
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()

        return dataset

    def _load_gt_npy(
        self, test_case: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground truth instance map and type map from .npy file

        Args:
            test_case (Union[str, Path]): Path to test case numpy array

        Returns:
            Tuple[np.ndarray, np.ndarray]: Instance map and type map
            * np.ndarray: Instance map ordered from 1 to num_nuclei in image, shape: H,W
            * np.ndarray: Type map, shape: H,W
        """
        # Load the .npy file as specified by user
        label = np.load(test_case, allow_pickle=True)
        
        # Extract inst_map and type_map from dictionary
        if isinstance(label, np.ndarray) and label.shape == ():
            # If it's a 0-d array containing a dict
            label = label.item()
        
        gt_inst_map = label.get("inst_map")
        gt_type_map = label.get("type_map")
        
        # Remap instance labels to be sequential
        gt_inst_map = remap_label(gt_inst_map, by_size=False)

        return gt_inst_map, gt_type_map

    def run_inference(
        self,
    ) -> None:
        """Run inference on the CRC CODEX dataset

        Loads CellViT model, classifier, and dataset, then performs inference and calculates metrics.
        """
        self.logger.info("Starting CRC CODEX Inference")
        
        # Load models
        self._load_cellvit_model()
        self._load_model()

        # Setup dataset and data loader
        inference_transforms = self.get_inference_transforms()
        self.inference_dataset = self._load_dataset(
            transforms=inference_transforms,
            normalize_stains=self.run_conf["preprocessing"].get("normalize_stains", False)
        )
        self.inference_dataloader = DataLoader(
            self.inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        self.logger.info(f"Length of dataset: {len(self.inference_dataset)}")

        # Setup postprocessor
        # Use CellViT's num_nuclei_classes for postprocessor
        cellvit_num_types = self.cellvit_model.num_nuclei_classes
        self.postprocessor = DetectionCellPostProcessorCupy(
            wsi=None, nr_types=cellvit_num_types
        )

        # Get label map
        self.label_map = self.inference_dataset.get_nuclei_types()
        self.logger.info(f"Label map: {self.label_map}")

        # Run inference loop
        self.logger.info("Running inference loop")
        self._run_inference_loop()

        # Calculate and log metrics
        self.logger.info("Calculating metrics")
        self._calculate_and_log_metrics()

        self.logger.info("Inference complete!")

    def _run_inference_loop(self) -> None:
        """Main inference loop"""
        self.step_results = []
        
        with torch.no_grad():
            for step_idx, batch in enumerate(
                tqdm.tqdm(self.inference_dataloader, desc="Processing images")
            ):
                # Get batch data
                images = batch["image"].to(self.device)
                inst_maps = batch["inst_map"].cpu().numpy()
                type_maps = batch["type_map"].cpu().numpy()
                image_names = batch["image_name"]

                # Run CellViT inference
                predictions = self.cellvit_model.forward(
                    images, retrieve_tokens=True
                )

                # Postprocess predictions
                instance_map, nuclei_dict = self.postprocessor.post_process_batch(
                    predictions, magnification=self.run_conf.get("magnification", 40)
                )

                # Update with classifier predictions
                nuclei_dict = self.update_cell_dict_with_predictions(
                    nuclei_dict, images
                )

                # Store results
                result = {
                    "image_name": image_names[0],
                    "predictions": nuclei_dict,
                    "gt_inst_map": inst_maps[0],
                    "gt_type_map": type_maps[0],
                    "pred_inst_map": instance_map[0],
                }
                self.step_results.append(result)

    def _calculate_and_log_metrics(self) -> None:
        """Calculate and log all metrics"""
        # Initialize metric accumulators
        all_metrics = {
            "dice": [],
            "aji": [],
            "aji_plus": [],
            "pq": [],
            "dq": [],
            "sq": [],
        }
        
        # Per-nuclei-type metrics
        nuclei_types = list(self.label_map.keys())
        nuclei_types.remove(0)  # Remove background
        
        per_type_metrics = {
            nt: {
                "dice": [],
                "aji": [],
                "pq": [],
                "dq": [],
                "sq": [],
                "f1_cell": [],
                "prec_cell": [],
                "rec_cell": [],
            }
            for nt in nuclei_types
        }

        # Calculate metrics for each image
        for result in tqdm.tqdm(self.step_results, desc="Calculating metrics"):
            gt_inst_map = result["gt_inst_map"]
            gt_type_map = result["gt_type_map"]
            pred_inst_map = result["pred_inst_map"]
            predictions = result["predictions"]

            # Create predicted type map
            pred_type_map = np.zeros_like(gt_type_map)
            for cell_id, cell_info in predictions.items():
                mask = pred_inst_map == cell_id
                pred_type_map[mask] = cell_info["type"]

            # Calculate binary metrics
            gt_binary = (gt_inst_map > 0).astype(np.uint8)
            pred_binary = (pred_inst_map > 0).astype(np.uint8)
            
            dice_score = get_dice_1(gt_binary, pred_binary)
            aji_score = get_fast_aji(gt_inst_map, pred_inst_map)
            aji_plus_score = get_fast_aji_plus(gt_inst_map, pred_inst_map)
            
            all_metrics["dice"].append(dice_score)
            all_metrics["aji"].append(aji_score)
            all_metrics["aji_plus"].append(aji_plus_score)

            # Calculate PQ metrics (overall)
            pq_metrics = get_fast_pq(gt_inst_map, pred_inst_map)
            all_metrics["pq"].append(pq_metrics[0])
            all_metrics["dq"].append(pq_metrics[1])
            all_metrics["sq"].append(pq_metrics[2])

            # Calculate per-type metrics
            for nuclei_type in nuclei_types:
                # Get binary masks for this type
                gt_type_binary = (gt_type_map == nuclei_type).astype(np.uint8)
                pred_type_binary = (pred_type_map == nuclei_type).astype(np.uint8)
                
                # Get instance maps for this type
                gt_type_inst = gt_inst_map * gt_type_binary
                pred_type_inst = pred_inst_map * pred_type_binary
                
                # Remap to sequential labels
                gt_type_inst = remap_label(gt_type_inst, by_size=False)
                pred_type_inst = remap_label(pred_type_inst, by_size=False)
                
                # Calculate metrics
                if gt_type_inst.max() > 0:  # Only if there are GT cells of this type
                    dice_type = get_dice_1(gt_type_binary, pred_type_binary)
                    aji_type = get_fast_aji(gt_type_inst, pred_type_inst)
                    pq_type = get_fast_pq(gt_type_inst, pred_type_inst)
                    
                    per_type_metrics[nuclei_type]["dice"].append(dice_type)
                    per_type_metrics[nuclei_type]["aji"].append(aji_type)
                    per_type_metrics[nuclei_type]["pq"].append(pq_type[0])
                    per_type_metrics[nuclei_type]["dq"].append(pq_type[1])
                    per_type_metrics[nuclei_type]["sq"].append(pq_type[2])
                    
                    # Cell-level detection metrics
                    gt_cells = np.unique(gt_type_inst)[1:]  # Exclude background
                    pred_cells = np.unique(pred_type_inst)[1:]
                    
                    tp = 0
                    fp = 0
                    fn = 0
                    
                    if len(gt_cells) > 0 and len(pred_cells) > 0:
                        # Pair cells
                        paired, unpaired_gt, unpaired_pred = pair_coordinates(
                            gt_type_inst, pred_type_inst, 15
                        )
                        tp = len(paired)
                        fn = len(unpaired_gt)
                        fp = len(unpaired_pred)
                    elif len(gt_cells) > 0:
                        fn = len(gt_cells)
                    elif len(pred_cells) > 0:
                        fp = len(pred_cells)
                    
                    # Calculate F1, precision, recall
                    if tp + fp + fn > 0:
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                        
                        per_type_metrics[nuclei_type]["f1_cell"].append(f1)
                        per_type_metrics[nuclei_type]["prec_cell"].append(prec)
                        per_type_metrics[nuclei_type]["rec_cell"].append(rec)

        # Log overall metrics
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Overall Metrics:")
        self.logger.info("=" * 70)
        for metric_name, metric_values in all_metrics.items():
            if len(metric_values) > 0:
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                self.logger.info(f"{metric_name.upper()}: {mean_val:.4f} ± {std_val:.4f}")

        # Log per-type metrics
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Per-Nuclei-Type Metrics:")
        self.logger.info("=" * 70)
        for nuclei_type in nuclei_types:
            type_name = self.label_map[nuclei_type]
            self.logger.info(f"\n{type_name} (Type {nuclei_type}):")
            self.logger.info("-" * 70)
            for metric_name, metric_values in per_type_metrics[nuclei_type].items():
                if len(metric_values) > 0:
                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values)
                    self.logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    self.logger.info(f"  {metric_name}: N/A (no samples)")

        # Save results to JSON
        results_dict = {
            "overall": {k: float(np.mean(v)) if len(v) > 0 else None for k, v in all_metrics.items()},
            "per_type": {
                self.label_map[nt]: {
                    k: float(np.mean(v)) if len(v) > 0 else None 
                    for k, v in metrics.items()
                }
                for nt, metrics in per_type_metrics.items()
            }
        }
        
        output_file = self.logdir / "inference_results.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        self.logger.info(f"\nResults saved to: {output_file}")

    @staticmethod
    def parse_arguments() -> dict:
        """Parse command line arguments

        Returns:
            dict: Parsed arguments as dictionary
        """
        parser = argparse.ArgumentParser(
            description="Run CRC CODEX inference with CellViT classifier",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained classifier",
            required=True,
        )
        parser.add_argument(
            "--cellvit_path",
            type=str,
            help="Path to the pretrained CellViT model (.pth file)",
            required=True,
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            help="Path to the CRC CODEX dataset root folder",
            required=True,
        )
        parser.add_argument(
            "--split",
            type=str,
            default="test",
            help="Dataset split to use (e.g., 'test', 'val')",
        )
        parser.add_argument(
            "--checkpoint_name",
            type=str,
            default="model_best.pth",
            help="Name of the checkpoint file in logdir/checkpoints/",
        )
        parser.add_argument(
            "--gpu", type=int, default=0, help="GPU ID to use"
        )
        parser.add_argument(
            "--magnification",
            type=int,
            default=40,
            help="Magnification level (20 or 40)",
        )

        args = parser.parse_args()
        return vars(args)


if __name__ == "__main__":
    # Parse arguments
    configuration = CellViTInfExpCRCCodex.parse_arguments()

    # Run inference
    inference_experiment = CellViTInfExpCRCCodex(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        split=configuration["split"],
        checkpoint_name=configuration["checkpoint_name"],
        gpu=configuration["gpu"],
    )
    
    # Set magnification if provided
    if "magnification" in configuration:
        inference_experiment.run_conf["magnification"] = configuration["magnification"]
    
    inference_experiment.run_inference()
