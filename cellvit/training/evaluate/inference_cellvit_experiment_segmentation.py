# -*- coding: utf-8 -*-
# Generic Nuclei Segmentation Dataset Inference Code
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artificial Intelligence in Medicine,
# University Medicine Essen

"""
Generic evaluation script for nuclei-only segmentation datasets.

⚠️  IMPORTANT: Use this script for custom nuclei segmentation datasets that:
    - Have a generic segmentation dataset structure (images/ and labels/ folders)
    - Contain nuclei type labels (no tissue types)
    - Use instance segmentation masks stored as .npy or .mat files
    - Have a label_map.yaml or dataset_config.yaml defining nuclei types

When to use this script:
    ✅ You have a custom nuclei segmentation dataset
    ✅ Your dataset structure is:
        dataset_path/
        ├── {split_name}/  # e.g., "test", "val", "Test", etc.
        │   ├── images/
        │   │   └── *.png, *.jpg
        │   └── labels/  # or "labels-1000-1000", etc.
        │       └── *.npy or *.mat files
        └── label_map.yaml  # or dataset_config.yaml
    ✅ Ground truth files contain:
        - inst_map: Instance segmentation map (H, W)
        - type_map: Cell type map (H, W)
    ✅ You want per-class F1, Precision, Recall metrics
    ✅ You want segmentation metrics (Dice, AJI, PQ)

Do NOT use this script if:
    ❌ Your dataset has tissue types (use inference_cellvit_experiment_pannuke.py)
    ❌ Your dataset is CoNSeP-specific (use inference_cellvit_experiment_consep.py)
    ❌ You have CSV-based detection annotations (use inference_cellvit_custom_classifier.py)

Output:
    This script generates:
    - Per-class F1, Precision, Recall scores for all nuclei types
    - Binary segmentation metrics (Dice, AJI, PQ)
    - Per-class PQ scores
    - Confusion matrix
    - Cell predictions as JSON files
    - Complete inference results in JSON format

For more information:
    - See docs/EVALUATION_GUIDE.md
    - See docs/SEGMENTATION_EVALUATION_QUICKSTART.md
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
import yaml
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
from cellvit.training.datasets.segmentation_dataset import SegmentationDataset
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


class CellViTInfExpNucleiSegmentation(CellViTClassifierInferenceExperiment):
    """Generic Nuclei Segmentation Dataset Inference Experiment"""

    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        dataset_path: Union[Path, str],
        split: str = "test",
        label_map_file: str = "label_map.yaml",
        gt_format: str = "npy",
        normalize_stains: bool = False,
        gpu: int = 0,
        checkpoint_name: str = "model_best.pth",
    ):
        """Initialize Generic Nuclei Segmentation Inference Experiment

        Args:
            logdir (Union[Path, str]): Path to the log directory with the trained head
            cellvit_path (Union[Path, str]): Path to the CellViT model
            dataset_path (Union[Path, str]): Path to the dataset root folder
            split (str, optional): Name of the split to evaluate (e.g., "test", "val", "Test"). Defaults to "test".
            label_map_file (str, optional): Name of the label map file (label_map.yaml or dataset_config.yaml). Defaults to "label_map.yaml".
            gt_format (str, optional): Format of ground truth files ("npy" or "mat"). Defaults to "npy".
            normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
            gpu (int, optional): CUDA GPU id to use. Defaults to 0.
            checkpoint_name (str, optional): Name of the checkpoint file. Defaults to "model_best.pth".
        """
        self.split = split
        self.label_map_file = label_map_file
        self.gt_format = gt_format.lower()
        assert self.gt_format in ["npy", "mat"], "gt_format must be 'npy' or 'mat'"

        super().__init__(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=normalize_stains,
            gpu=gpu,
            checkpoint_name=checkpoint_name,
        )

        self._validate_dataset_structure()
        self._load_label_map()

    def _validate_dataset_structure(self) -> None:
        """Validate that the dataset has the expected structure
        
        Raises:
            FileNotFoundError: If dataset structure is invalid
        """
        errors = []
        
        # Check if split folder exists
        split_path = self.dataset_path / self.split
        if not split_path.exists():
            errors.append(
                f"❌ Split folder not found: {split_path}\n"
                f"   Expected: {self.dataset_path}/{self.split}/\n"
                f"   Available splits:\n"
            )
            if self.dataset_path.exists():
                subdirs = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
                if subdirs:
                    for subdir in subdirs:
                        errors.append(f"      - {subdir}")
                else:
                    errors.append(f"      (no subdirectories found)")
            
        # Check if images folder exists
        images_path = split_path / "images"
        if not images_path.exists():
            errors.append(
                f"❌ Images folder not found: {images_path}\n"
                f"   Expected: {split_path}/images/\n"
            )
        
        # Check if labels folder exists
        label_path = self._get_gt_label_folder()
        if not label_path.exists():
            errors.append(
                f"❌ Labels folder not found\n"
                f"   Checked these locations:\n"
                f"      - {split_path}/labels/\n"
                f"      - {split_path}/Labels/\n"
                f"      - {split_path}/labels-1000-1000/\n"
                f"      - {split_path}/annotations/\n"
                f"   Make sure your dataset has a labels folder in the split directory.\n"
            )
        
        if errors:
            error_message = (
                "\n" + "="*70 + "\n"
                "🛑 Dataset Structure Validation Failed\n"
                "="*70 + "\n\n"
                + "\n\n".join(errors) + "\n\n"
                + "="*70 + "\n"
                "Expected dataset structure:\n"
                f"{self.dataset_path}/\n"
                f"├── {self.split}/\n"
                f"│   ├── images/\n"
                f"│   │   └── *.png, *.jpg\n"
                f"│   └── labels/  # or labels-1000-1000, etc.\n"
                f"│       └── *.{self.gt_format}\n"
                f"└── {self.label_map_file}\n\n"
                + "="*70 + "\n"
                "See docs/HOW_TO_RUN_SEGMENTATION_EVALUATION.md for more details.\n"
                + "="*70 + "\n"
            )
            raise FileNotFoundError(error_message)
        
        self.logger.info(f"✓ Dataset structure validated successfully")
        self.logger.info(f"  Split: {self.split}")
        self.logger.info(f"  Images: {images_path}")
        self.logger.info(f"  Labels: {label_path}")


    def _load_label_map(self) -> None:
        """Load nuclei type labels from label_map.yaml or dataset_config.yaml"""
        label_map_path = self.dataset_path / self.label_map_file

        if not label_map_path.exists():
            self.logger.warning(
                f"Label map file not found at {label_map_path}. "
                f"Using default nuclei type names."
            )
            self.nuclei_type_names = {
                i: f"Type_{i}" for i in range(self.num_classes)
            }
            return

        with open(label_map_path, "r") as f:
            label_data = yaml.safe_load(f)

        # Handle both direct mapping (1: "Type1") and nested structure
        if isinstance(label_data, dict):
            # Convert string keys to int and create 0-indexed dict
            self.nuclei_type_names = {}
            for k, v in label_data.items():
                try:
                    idx = int(k) - 1
                    if idx >= 0:
                        self.nuclei_type_names[idx] = v
                except (ValueError, TypeError):
                    self.logger.warning(f"Skipping invalid label key: {k}")
        else:
            self.logger.warning(
                f"Unexpected label map format in {label_map_path}. "
                f"Using default nuclei type names."
            )
            self.nuclei_type_names = {
                i: f"Type_{i}" for i in range(self.num_classes)
            }

        self.logger.info(f"Loaded nuclei type labels: {self.nuclei_type_names}")

    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load Generic Segmentation Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization should be applied

        Returns:
            Dataset: Segmentation Dataset
        """
        dataset = SegmentationDataset(
            dataset_path=self.dataset_path,
            split=self.split,
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()

        return dataset

    def _get_gt_label_folder(self) -> Path:
        """Get the ground truth label folder path

        Returns:
            Path: Path to the label folder
        """
        # Try common label folder names
        possible_names = [
            "labels",
            "Labels",
            "labels-1000-1000",
            "Labels-1000-1000",
            "annotations",
            "Annotations",
        ]

        for name in possible_names:
            label_path = self.dataset_path / self.split / name
            if label_path.exists():
                return label_path

        # Default to "labels"
        return self.dataset_path / self.split / "labels"

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
        gt = np.load(test_case, allow_pickle=True)
        gt_inst_map = gt.item()["inst_map"]
        gt_inst_map = remap_label(gt_inst_map, by_size=False)
        gt_type_map = gt.item()["type_map"]

        return gt_inst_map, gt_type_map

    def _load_gt_mat(
        self, test_case: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground truth instance map and type map from .mat file

        Args:
            test_case (Union[str, Path]): Path to test case mat array

        Returns:
            Tuple[np.ndarray, np.ndarray]: Instance map and type map
            * np.ndarray: Instance map ordered from 1 to num_nuclei in image, shape: H,W
            * np.ndarray: Type map, shape: H,W
        """
        gt = loadmat(test_case)
        gt_inst_map = gt["inst_map"]
        gt_inst_map = remap_label(gt_inst_map, by_size=False)
        gt_type_map = gt["type_map"]

        return gt_inst_map, gt_type_map

    def _load_pred_map(
        self, cells: dict, img_shape: Tuple[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load prediction map from image cell dictionary

        Args:
            cells (dict): Cell dictionary for the image
            img_shape (Tuple[int]): Shape in Format (H, W)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
            * np.ndarray: Prediction map with instance ordering and types as first axis.
                Shape: Num_classes+1, H, W
            * np.ndarray: Instance map ordered from 1 to num_nuclei in image, shape: H,W
            * np.ndarray: Type map, shape: H,W
        """
        pred_inst_map = np.zeros(img_shape, dtype=np.int32)
        pred_class_map = np.zeros(img_shape, dtype=np.int32)

        for cell_id, cell_data in cells.items():
            contour = np.array(cell_data["contour"])
            cell_type = cell_data["type"]
            contour = contour.reshape((-1, 1, 2))
            contour = np.vstack((contour, [contour[0]]))
            cell_id = int(cell_id)
            cv2.fillPoly(pred_inst_map, [contour], cell_id)
            cv2.fillPoly(pred_class_map, [contour], cell_type + 1)

        # Convert to multi-class format
        pred_map = np.zeros((self.num_classes + 1, *img_shape), dtype=np.int32)
        for class_idx in range(1, self.num_classes + 1):
            mask = pred_class_map == class_idx
            pred_map[class_idx][mask] = pred_inst_map[mask]

        return pred_map, pred_inst_map, pred_class_map

    def _get_global_classifier_scores(
        self, predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate global metrics for the classification head

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            probabilities (torch.Tensor): Probabilities for all classes. Shape: Num-cells x Num-classes
            gt (torch.Tensor): Ground-truth Predictions. Shape: Num-cells

        Returns:
            Tuple[float, float, float, float, float, float]:
                * F1-Score
                * Precision
                * Recall
                * Accuracy
                * Auroc
                * AP
        """
        auroc_func = AUROC(task="multiclass", num_classes=self.num_classes)
        acc_func = Accuracy(task="multiclass", num_classes=self.num_classes)
        f1_func = F1Score(task="multiclass", num_classes=self.num_classes)
        prec_func = Precision(task="multiclass", num_classes=self.num_classes)
        recall_func = Recall(task="multiclass", num_classes=self.num_classes)
        average_prec_func = AveragePrecision(
            task="multiclass", num_classes=self.num_classes
        )

        auroc_score = float(auroc_func(probabilities, gt).detach().cpu())
        acc_score = float(acc_func(predictions, gt).detach().cpu())
        f1_score = float(f1_func(predictions, gt).detach().cpu())
        prec_score = float(prec_func(predictions, gt).detach().cpu())
        recall_score = float(recall_func(predictions, gt).detach().cpu())
        average_prec = float(average_prec_func(probabilities, gt).detach().cpu())

        return f1_score, prec_score, recall_score, acc_score, auroc_score, average_prec

    def _plot_confusion_matrix(
        self,
        predictions: torch.Tensor,
        gt: torch.Tensor,
        test_result_dir: Union[Path, str],
    ) -> None:
        """Plot and save the confusion matrix (normalized and non-normalized)

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            gt (torch.Tensor): Ground-truth Predictions. Shape: Num-cells
            test_result_dir (Union[Path, str]): Path to the test result directory
        """
        conf_matrix = pycm.ConfusionMatrix(
            actual_vector=gt.detach().cpu().numpy(),
            predict_vector=predictions.detach().cpu().numpy(),
        )
        conf_matrix.relabel(self.nuclei_type_names)
        conf_matrix.save_stat(
            str(test_result_dir / "confusion_matrix_summary"), summary=True
        )

        axs = conf_matrix.plot(
            cmap=plt.cm.Blues,
            plot_lib="seaborn",
            title="Confusion-Matrix",
            number_label=True,
        )
        fig = axs.get_figure()
        fig.savefig(str(test_result_dir / "confusion_matrix.png"), dpi=600)
        fig.savefig(str(test_result_dir / "confusion_matrix.pdf"), dpi=600)
        plt.close(fig)

        axs = conf_matrix.plot(
            cmap=plt.cm.Blues,
            plot_lib="seaborn",
            title="Confusion-Matrix",
            number_label=True,
            normalized=True,
        )
        fig = axs.get_figure()
        fig.savefig(str(test_result_dir / "confusion_matrix_normalized.png"), dpi=600)
        fig.savefig(str(test_result_dir / "confusion_matrix_normalized.pdf"), dpi=600)
        plt.close(fig)

    def _calculate_pipeline_scores(self, cell_dict: dict) -> Tuple[dict, dict, dict]:
        """Calculate the final pipeline scores

        Args:
            cell_dict (dict): Cell dictionary

        Returns:
            Tuple[dict, dict, dict]: Segmentation, PQ and Detection Scores
        """
        self.logger.info("Calculating dataset scores")

        segmentation_scores = {
            "binary": {
                "dice": [],
                "fast_aji": [],
                "fast_aji_plus": [],
            }
        }
        pq_scores = {
            "binary": {
                "pq": [],
                "dq": [],
                "sq": [],
            },
            "mean": {
                "pq": [],
                "dq": [],
                "sq": [],
            },
            "mean+": {
                "pq": [],
                "dq": [],
                "sq": [],
            },
        }
        detection_tracker = {
            "paired_all": [],
            "unpaired_true_all": [],
            "unpaired_pred_all": [],
            "true_inst_type_all": [],
            "pred_inst_type_all": [],
        }
        true_idx_offset = 0
        pred_idx_offset = 0
        mpq_info_list = []

        gt_label_folder = self._get_gt_label_folder()

        for image_idx, (image_name, cells) in tqdm.tqdm(
            enumerate(cell_dict.items()), total=len(cell_dict)
        ):
            # Load ground truth
            gt_file_ext = "npy" if self.gt_format == "npy" else "mat"
            gt_file = gt_label_folder / f"{image_name}.{gt_file_ext}"

            if not gt_file.exists():
                self.logger.warning(f"Ground truth file not found: {gt_file}")
                continue

            if self.gt_format == "npy":
                gt_inst_map, gt_type_map = self._load_gt_npy(gt_file)
            else:
                gt_inst_map, gt_type_map = self._load_gt_mat(gt_file)

            pred_map, pred_inst_map, pred_type_map = self._load_pred_map(
                cells, img_shape=gt_inst_map.shape
            )

            pred_inst_map_binary = remap_label(
                binarize(pred_map.transpose(1, 2, 0)), by_size=False
            )

            # Segmentation scores
            dice_1 = get_dice_1(true=gt_inst_map, pred=pred_inst_map_binary)
            aji = get_fast_aji(true=gt_inst_map, pred=pred_inst_map_binary)
            aji_plus = get_fast_aji_plus(true=gt_inst_map, pred=pred_inst_map_binary)
            segmentation_scores["binary"]["dice"].append(dice_1)
            segmentation_scores["binary"]["fast_aji"].append(aji)
            segmentation_scores["binary"]["fast_aji_plus"].append(aji_plus)

            # Panoptic scores
            (dq, sq, pq), _ = get_fast_pq(true=gt_inst_map, pred=pred_inst_map_binary)
            pq_scores["binary"]["pq"].append(pq)
            pq_scores["binary"]["dq"].append(dq)
            pq_scores["binary"]["sq"].append(sq)

            # Per cell type scores
            image_pq = []
            pq_clx = {"dq": [], "sq": [], "pq": []}
            for cell_type_idx in range(1, self.num_classes + 1):
                pred_nuclei_inst_map = remap_label(
                    pred_map[cell_type_idx, :, :], by_size=False
                )

                gt_nuclei_inst_map = gt_inst_map * (gt_type_map == cell_type_idx)
                gt_nuclei_inst_map = remap_label(gt_nuclei_inst_map, by_size=False)

                pq_oneclass_info = get_pq(
                    gt_nuclei_inst_map, pred_nuclei_inst_map, remap=False
                )
                if len(np.unique(gt_nuclei_inst_map)) == 1:
                    pq_clx["dq"].append(np.nan)
                    pq_clx["sq"].append(np.nan)
                    pq_clx["pq"].append(np.nan)
                else:
                    pq_clx["dq"].append(pq_oneclass_info[0][0])
                    pq_clx["sq"].append(pq_oneclass_info[0][1])
                    pq_clx["pq"].append(pq_oneclass_info[0][2])

                image_pq.append(pq_oneclass_info)

            pq_scores["mean"]["dq"].append(np.nanmean(pq_clx["dq"]))
            pq_scores["mean"]["sq"].append(np.nanmean(pq_clx["sq"]))
            pq_scores["mean"]["pq"].append(np.nanmean(pq_clx["pq"]))

            mpq_info = []
            for single_pq in image_pq:
                tp = single_pq[1][0]
                fp = single_pq[1][1]
                fn = single_pq[1][2]
                sum_iou = single_pq[2]
                mpq_info.append([tp, fp, fn, sum_iou])
            mpq_info_list.append(mpq_info)

            # Detection scores
            gt_inst_map_tensor = torch.Tensor(gt_inst_map).unsqueeze(0)
            gt_type_map_tensor = torch.Tensor(gt_type_map).unsqueeze(0)
            gt_type_map_oh = F.one_hot(
                gt_type_map_tensor.to(torch.int64), self.num_classes + 1
            ).type(torch.float32)
            gt_type_map_oh = gt_type_map_oh.permute(0, 3, 1, 2)[:, 1:, :, :]
            gt_instance_types = calculate_instances(
                torch.Tensor(gt_type_map_oh), torch.Tensor(gt_inst_map_tensor)
            )
            true_centroids = np.array(
                [v["centroid"] for k, v in gt_instance_types[0].items()]
            )
            true_instance_type = np.array(
                [v["type"] for k, v in gt_instance_types[0].items()]
            )

            # Recalculate cell dict items
            pred_instance_types_rescaled = calculate_instances(
                torch.Tensor(np.clip(pred_map[1:, ...], 0, 1)[None, ...]),
                torch.Tensor(pred_inst_map)[None, :],
            )
            pred_centroids = np.array(
                [v["centroid"] for k, v in pred_instance_types_rescaled[0].items()]
            )
            pred_instance_type = np.array(
                [v["type"] for k, v in pred_instance_types_rescaled[0].items()]
            )

            if true_centroids.shape[0] == 0:
                true_centroids = np.array([[0, 0]])
                true_instance_type = np.array([0])
            if pred_centroids.shape[0] == 0:
                pred_centroids = np.array([[0, 0]])
                pred_instance_type = np.array([0])

            pairing_radius = 12
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroids, pred_centroids, pairing_radius
            )
            true_idx_offset = (
                true_idx_offset + detection_tracker["true_inst_type_all"][-1].shape[0]
                if image_idx != 0
                else 0
            )
            pred_idx_offset = (
                pred_idx_offset + detection_tracker["pred_inst_type_all"][-1].shape[0]
                if image_idx != 0
                else 0
            )
            detection_tracker["true_inst_type_all"].append(true_instance_type)
            detection_tracker["pred_inst_type_all"].append(pred_instance_type)

            if paired.shape[0] != 0:
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                detection_tracker["paired_all"].append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            detection_tracker["unpaired_true_all"].append(unpaired_true)
            detection_tracker["unpaired_pred_all"].append(unpaired_pred)

        # Calculate mean+ PQ scores
        mpq_info_metrics = np.array(mpq_info_list, dtype="float")
        total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)
        mdq_list = []
        msq_list = []
        mpq_list = []
        for cat_idx in range(total_mpq_info_metrics.shape[0]):
            total_tp = total_mpq_info_metrics[cat_idx][0]
            total_fp = total_mpq_info_metrics[cat_idx][1]
            total_fn = total_mpq_info_metrics[cat_idx][2]
            total_sum_iou = total_mpq_info_metrics[cat_idx][3]
            dq = total_tp / ((total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6)
            sq = total_sum_iou / (total_tp + 1.0e-6)
            mdq_list.append(dq)
            msq_list.append(sq)
            mpq_list.append(dq * sq)

        pq_scores["mean+"]["dq"] = mdq_list
        pq_scores["mean+"]["sq"] = msq_list
        pq_scores["mean+"]["pq"] = mpq_list

        # Concatenate detection tracker arrays
        detection_tracker["paired_all"] = np.concatenate(
            detection_tracker["paired_all"], axis=0
        )
        detection_tracker["unpaired_true_all"] = np.concatenate(
            detection_tracker["unpaired_true_all"], axis=0
        )
        detection_tracker["unpaired_pred_all"] = np.concatenate(
            detection_tracker["unpaired_pred_all"], axis=0
        )
        detection_tracker["true_inst_type_all"] = np.concatenate(
            detection_tracker["true_inst_type_all"], axis=0
        )
        detection_tracker["pred_inst_type_all"] = np.concatenate(
            detection_tracker["pred_inst_type_all"], axis=0
        )

        detection_tracker["paired_true_type"] = detection_tracker["true_inst_type_all"][
            detection_tracker["paired_all"][:, 0]
        ]
        detection_tracker["paired_pred_type"] = detection_tracker["pred_inst_type_all"][
            detection_tracker["paired_all"][:, 1]
        ]
        detection_tracker["unpaired_true_type"] = detection_tracker[
            "true_inst_type_all"
        ][detection_tracker["unpaired_true_all"]]
        detection_tracker["unpaired_pred_type"] = detection_tracker[
            "pred_inst_type_all"
        ][detection_tracker["unpaired_pred_all"]]

        # Global detection scores
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=detection_tracker["paired_true_type"],
            paired_pred=detection_tracker["paired_pred_type"],
            unpaired_true=detection_tracker["unpaired_true_type"],
            unpaired_pred=detection_tracker["unpaired_pred_type"],
        )
        detection_scores = {"binary": {}, "cell_types": {}}
        detection_scores["binary"] = {"f1": f1_d, "prec": prec_d, "rec": rec_d}

        # Per-class detection scores
        for cell_idx in range(self.num_classes):
            detection_scores["cell_types"][cell_idx] = {}
            f1_c, prec_c, rec_c = cell_type_detection_scores(
                paired_true=detection_tracker["paired_true_type"],
                paired_pred=detection_tracker["paired_pred_type"],
                unpaired_true=detection_tracker["unpaired_true_type"],
                unpaired_pred=detection_tracker["unpaired_pred_type"],
                type_id=cell_idx,
            )
            detection_scores["cell_types"][cell_idx] = {
                "f1": f1_c,
                "prec": prec_c,
                "rec": rec_c,
            }

        # Calculate means
        segmentation_scores["binary"]["dice"] = np.nanmean(
            segmentation_scores["binary"]["dice"]
        )
        segmentation_scores["binary"]["fast_aji"] = np.nanmean(
            segmentation_scores["binary"]["fast_aji"]
        )
        segmentation_scores["binary"]["fast_aji_plus"] = np.nanmean(
            segmentation_scores["binary"]["fast_aji_plus"]
        )

        pq_scores["binary"]["pq"] = np.nanmean(pq_scores["binary"]["pq"])
        pq_scores["binary"]["dq"] = np.nanmean(pq_scores["binary"]["dq"])
        pq_scores["binary"]["sq"] = np.nanmean(pq_scores["binary"]["sq"])
        pq_scores["mean"]["pq"] = np.nanmean(pq_scores["mean"]["pq"])
        pq_scores["mean"]["dq"] = np.nanmean(pq_scores["mean"]["dq"])
        pq_scores["mean"]["sq"] = np.nanmean(pq_scores["mean"]["sq"])

        pq_scores["cell_types+"] = {}
        for cell_idx, _ in enumerate(pq_scores["mean+"]["pq"]):
            pq_scores["cell_types+"][cell_idx] = {}
            pq_scores["cell_types+"][cell_idx]["pq"] = pq_scores["mean+"]["pq"][
                cell_idx
            ]
            pq_scores["cell_types+"][cell_idx]["dq"] = pq_scores["mean+"]["dq"][
                cell_idx
            ]
            pq_scores["cell_types+"][cell_idx]["sq"] = pq_scores["mean+"]["sq"][
                cell_idx
            ]

        pq_scores["mean+"]["pq"] = np.nanmean(pq_scores["mean+"]["pq"])
        pq_scores["mean+"]["dq"] = np.nanmean(pq_scores["mean+"]["dq"])
        pq_scores["mean+"]["sq"] = np.nanmean(pq_scores["mean+"]["sq"])

        return segmentation_scores, pq_scores, detection_scores

    def update_cell_dict_with_predictions(
        self,
        cell_dict: dict,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        metadata: List[Tuple[float, float, str]],
    ) -> dict:
        """Update the cell dictionary with the predictions from the classifier

        Args:
            cell_dict (dict): Cell dictionary with CellViT default predictions
            predictions (np.ndarray): Classifier predictions of the class
            probabilities (np.ndarray): Classifier output probabilities
            metadata (List[Tuple[float, float, str]]): Cell metadata

        Returns:
            dict: Updated cell dictionary
        """
        self.logger.info("Updating cell predictions with classifier outputs")
        for pred, prob, inform in tqdm.tqdm(
            zip(predictions, probabilities, metadata), total=len(predictions)
        ):
            cell_found = False
            image_name = inform[2]
            image_cell_dict = cell_dict[image_name]
            row_pred, col_pred = inform[:2]
            row_pred = float(f"{row_pred:.0f}")
            col_pred = float(f"{col_pred:.0f}")

            for cell_idx, properties in image_cell_dict.items():
                row, col = properties["centroid"]
                row = float(f"{row:.0f}")
                col = float(f"{col:.0f}")
                if row == row_pred and col == col_pred:
                    cell_dict[image_name][cell_idx]["type"] = int(pred)
                    cell_dict[image_name][cell_idx]["type_prob"] = float(
                        prob[int(pred)]
                    )
                    cell_dict[image_name][cell_idx]["bbox"] = cell_dict[image_name][
                        cell_idx
                    ]["bbox"].tolist()
                    cell_dict[image_name][cell_idx]["centroid"] = cell_dict[image_name][
                        cell_idx
                    ]["centroid"].tolist()
                    cell_dict[image_name][cell_idx]["contour"] = cell_dict[image_name][
                        cell_idx
                    ]["contour"].tolist()
                    cell_found = True
            assert cell_found, f"Cell not found for centroid ({row_pred}, {col_pred}) in image {image_name}"

        return cell_dict

    def run_inference(self):
        """Run Inference on Test Dataset for Generic Nuclei Segmentation"""
        extracted_cells = []
        extracted_cells_cleaned = []
        image_pred_dict = {}
        detection_scores = {
            "F1": [],
            "Prec": [],
            "Rec": [],
        }
        scores = {}

        postprocessor = DetectionCellPostProcessorCupy(
            wsi=None, nr_types=self.num_classes + 1
        )
        cellvit_dl = DataLoader(
            self.inference_dataset,
            batch_size=4,
            num_workers=8,
            shuffle=False,
            collate_fn=self.inference_dataset.collate_batch,
        )

        # Step 1: Extract cells with CellViT
        self.logger.info("Step 1: Extracting cells with CellViT")
        with torch.no_grad():
            for _, (images, cell_gt_batch, types_batch, image_names) in tqdm.tqdm(
                enumerate(cellvit_dl), total=len(cellvit_dl)
            ):
                (
                    batch_cells_cleaned,
                    batch_cells,
                    batch_pred_dict,
                    batch_f1s,
                    batch_recs,
                    batch_precs,
                ) = self._get_cellvit_result(
                    images=images,
                    cell_gt_batch=cell_gt_batch,
                    types_batch=types_batch,
                    image_names=image_names,
                    postprocessor=postprocessor,
                )
                extracted_cells = extracted_cells + batch_cells
                extracted_cells_cleaned = extracted_cells_cleaned + batch_cells_cleaned
                image_pred_dict.update(batch_pred_dict)
                detection_scores["F1"] = detection_scores["F1"] + batch_f1s
                detection_scores["Prec"] = detection_scores["Prec"] + batch_precs
                detection_scores["Rec"] = detection_scores["Rec"] + batch_recs

            cellvit_detection_scores = {
                "F1": float(np.mean(np.array(detection_scores["F1"]))),
                "Prec": float(np.mean(np.array(detection_scores["Prec"]))),
                "Rec": float(np.mean(np.array(detection_scores["Rec"]))),
            }
            self.logger.info(
                f"Extraction detection metrics - F1: {cellvit_detection_scores['F1']:.3f}, "
                f"Precision: {cellvit_detection_scores['Prec']:.3f}, "
                f"Recall: {cellvit_detection_scores['Rec']:.3f}"
            )
            scores["cellvit_scores"] = cellvit_detection_scores

        # Step 2: Classify Cell Tokens with the classifier
        self.logger.info("Step 2: Classifying cells with classifier")
        cleaned_inference_results = self._get_classifier_result(extracted_cells_cleaned)

        scores["classifier"] = {}
        (
            f1_score,
            prec_score,
            recall_score,
            acc_score,
            auroc_score,
            ap_score,
        ) = self._get_global_classifier_scores(
            predictions=cleaned_inference_results["predictions"],
            probabilities=cleaned_inference_results["probabilities"],
            gt=cleaned_inference_results["gt"],
        )
        self.logger.info(
            "Global Classifier Scores (without detection quality):"
        )
        self.logger.info(
            f"F1: {f1_score:.3f} - Prec: {prec_score:.3f} - Rec: {recall_score:.3f} - "
            f"Acc: {acc_score:.3f} - Auroc: {auroc_score:.3f} - AP: {ap_score:.3f}"
        )
        scores["classifier"]["global"] = {
            "F1": f1_score,
            "Prec": prec_score,
            "Rec": recall_score,
            "Acc": acc_score,
            "Auroc": auroc_score,
            "AP": ap_score,
        }
        self._plot_confusion_matrix(
            predictions=cleaned_inference_results["predictions"],
            gt=cleaned_inference_results["gt"],
            test_result_dir=self.test_result_dir,
        )

        # Step 3: Update predictions with uncleaned version
        self.logger.info("Step 3: Updating cell predictions")
        inference_results = self._get_classifier_result(extracted_cells)
        inference_results.pop("gt")
        cell_pred_dict = self.update_cell_dict_with_predictions(
            cell_dict=image_pred_dict,
            predictions=inference_results["predictions"].numpy(),
            probabilities=inference_results["probabilities"].numpy(),
            metadata=inference_results["metadata"],
        )

        # Store predictions as JSON
        (self.test_result_dir / "cell_predictions").mkdir(exist_ok=True)
        for image_name, cell_dict in cell_pred_dict.items():
            cell_dict = {int(k): v for k, v in cell_dict.items()}
            cell_dict = dict(sorted(cell_dict.items()))
            with open(
                self.test_result_dir / "cell_predictions" / f"{image_name}.json", "w"
            ) as json_file:
                json.dump(cell_dict, json_file, indent=2)

        # Step 4: Evaluate the whole pipeline
        self.logger.info("Step 4: Calculating final pipeline scores")
        (
            segmentation_scores,
            pq_scores,
            detection_scores,
        ) = self._calculate_pipeline_scores(cell_pred_dict)
        scores["pipeline"] = {
            "segmentation_scores": segmentation_scores,
            "detection_scores": detection_scores,
            "pq_scores": pq_scores,
        }

        # Replace cell_type indices by names
        scores["pipeline"]["pq_scores"]["cell_types+"] = {
            self.nuclei_type_names.get(k, f"Type_{k}"): v
            for k, v in scores["pipeline"]["pq_scores"]["cell_types+"].items()
        }
        scores["pipeline"]["detection_scores"]["cell_types"] = {
            self.nuclei_type_names.get(k, f"Type_{k}"): v
            for k, v in scores["pipeline"]["detection_scores"]["cell_types"].items()
        }

        scores_json = json.dumps(scores, indent=2)
        self.logger.info(f"{50*'*'}")
        self.logger.info("Final Results:")
        self.logger.info(scores_json)

        with open(self.test_result_dir / "inference_results.json", "w") as json_file:
            json.dump(scores, json_file, indent=2)

        self.logger.info(f"Results saved to: {self.test_result_dir}")


class CellViTInfExpNucleiSegmentationParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for generic nuclei segmentation datasets",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            required=True,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            required=True,
            help="Path to the nuclei segmentation dataset root folder",
        )
        parser.add_argument(
            "--cellvit_path",
            type=str,
            required=True,
            help="Path to the CellViT model checkpoint",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="test",
            help="Name of the split to evaluate (e.g., 'test', 'val', 'Test')",
        )
        parser.add_argument(
            "--label_map_file",
            type=str,
            default="label_map.yaml",
            help="Name of the label map file (label_map.yaml or dataset_config.yaml)",
        )
        parser.add_argument(
            "--gt_format",
            type=str,
            default="npy",
            choices=["npy", "mat"],
            help="Format of ground truth files ('npy' or 'mat')",
        )
        parser.add_argument(
            "--checkpoint_name",
            type=str,
            default="model_best.pth",
            help="Name of the checkpoint. Either 'model_best.pth', 'latest_checkpoint.pth' "
            "or one of the intermediate checkpoint names, e.g., 'checkpoint_100.pth'",
        )
        parser.add_argument(
            "--normalize_stains",
            action="store_true",
            help="If stains should be normalized for inference",
        )
        parser.add_argument(
            "--gpu", type=int, help="Number of CUDA GPU to use", default=0
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpNucleiSegmentationParser()
    configuration = configuration_parser.parse_arguments()

    experiment_inferer = CellViTInfExpNucleiSegmentation(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        split=configuration["split"],
        label_map_file=configuration["label_map_file"],
        gt_format=configuration["gt_format"],
        normalize_stains=configuration["normalize_stains"],
        gpu=configuration["gpu"],
        checkpoint_name=configuration["checkpoint_name"],
    )
    experiment_inferer.run_inference()
