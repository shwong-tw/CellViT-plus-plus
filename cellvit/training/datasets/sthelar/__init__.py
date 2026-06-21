# -*- coding: utf-8 -*-
# STHELAR Dataset for Cell Classifier Training
#
# Dataset: https://huggingface.co/datasets/FelicieGS/STHELAR_40x
# Paper: STHELAR (Scientific Data, 2026)
#
# This dataset class loads preprocessed STHELAR data (output of prepare_sthelar.py)
# in the format expected by CellViT++ classifier head training.

import csv
import json
from pathlib import Path
from typing import Callable, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchstain
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class STHELARDataset(Dataset):
    """STHELAR Dataset for CellViT++ Cell Classifier Training.

    Loads preprocessed STHELAR patches with cell detection annotations.
    Compatible with the CellViTHeadTrainer training loop.

    Expected directory structure (output of prepare_sthelar.py):
        dataset_path/
        ├── train/
        │   ├── images/     (*.png, 256x256 RGB H&E patches)
        │   └── detections/ (*.json, cell centroids + types)
        ├── val/
        │   ├── images/
        │   └── detections/
        └── test/
            ├── images/
            └── detections/

    Each detection JSON file contains a list of dicts:
        [{"centroid": [x, y], "type": int}, ...]

    Args:
        dataset_path (Union[Path, str]): Path to the preprocessed STHELAR dataset
        split (str): Dataset split ("train", "val", or "test")
        filelist_path (Union[Path, str], optional): CSV file listing patch names to use.
            If None, all patches in the split directory are used.
        transforms (Callable, optional): Albumentations transforms pipeline.
        normalize_stains (bool, optional): Whether to apply Macenko stain normalization.
        num_classes (int, optional): Number of cell type classes (excluding background).
            Used for validation. Default: 4 (5-class setup: 4 types + background).
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        filelist_path: Union[Path, str] = None,
        transforms: Callable = A.Compose(
            [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        ),
        normalize_stains: bool = False,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.normalize_stains = normalize_stains
        self.num_classes = num_classes
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split / "images"
        self.annotation_path = self.dataset_path / self.split / "detections"

        # Verify paths exist
        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_path}. "
                f"Run prepare_sthelar.py first to preprocess the dataset."
            )
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Detection directory not found: {self.annotation_path}. "
                f"Run prepare_sthelar.py first to preprocess the dataset."
            )

        # Get list of all images
        self.images = sorted(self.image_path.glob("*.png"))

        # Filter by filelist if provided
        if filelist_path is not None:
            selected_files = []
            with open(filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:  # skip empty lines
                        selected_files.append(row[0].strip())
            self.images = [f for f in self.images if f.stem in selected_files]

        # Map images to annotation files
        self.annotations = []
        for img_path in self.images:
            self.annotations.append(self.annotation_path / f"{img_path.stem}.json")

        # Caches for in-memory loading
        self.cache_images = {}
        self.cache_annotations = {}

        # Label map (will be populated from dataset_config.yaml if available)
        self._load_label_map()

    def _load_label_map(self) -> None:
        """Load label map from dataset config if available."""
        config_path = self.dataset_path / "dataset_config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if "label_map" in config:
                self.label_map = {int(k): v for k, v in config["label_map"].items()}
            else:
                self.label_map = {i: f"Class_{i}" for i in range(self.num_classes)}
        else:
            # Default 5-class label map
            self.label_map = {
                0: "Immune",
                1: "Stromal",
                2: "Epithelial",
                3: "Other",
            }

    def cache_dataset(self) -> None:
        """Cache the entire dataset in memory for faster training.

        Loads all images and annotations into dictionaries keyed by patch name.
        Recommended for datasets that fit in RAM.
        """
        for img_path, annot_path in tqdm.tqdm(
            zip(self.images, self.annotations),
            total=len(self.images),
            desc=f"Caching STHELAR {self.split}",
        ):
            img = Image.open(img_path).convert("RGB")
            self.cache_images[img_path.stem] = img

            if annot_path.exists():
                with open(annot_path, "r") as f:
                    cell_annot = json.load(f)
            else:
                cell_annot = []
            self.cache_annotations[img_path.stem] = cell_annot

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, list, list, str]:
        """Get a single sample from the dataset.

        Args:
            index: Sample index

        Returns:
            Tuple of:
                - image: (3, H, W) tensor, normalized
                - detections: List of (x, y) centroid coordinates
                - types: List of integer cell type labels (0-indexed)
                - img_name: Patch identifier string
        """
        img_path = self.images[index]
        img_name = img_path.stem

        # Load from cache or disk
        if img_name in self.cache_images:
            img = self.cache_images[img_name]
            cell_annot = self.cache_annotations[img_name]
        else:
            img = Image.open(img_path).convert("RGB")
            annot_path = self.annotations[index]
            if annot_path.exists():
                with open(annot_path, "r") as f:
                    cell_annot = json.load(f)
            else:
                cell_annot = []

        # Extract detections and types
        # Types stored as 1-indexed in JSON (matching CoNSeP convention), convert to 0-indexed
        detections = [(int(v["centroid"][0]), int(v["centroid"][1])) for v in cell_annot]
        types = [int(v["type"]) - 1 for v in cell_annot]

        # Stain normalization (optional)
        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))

        # Convert to numpy for albumentations
        img = np.array(img).astype(np.uint8)

        # Apply transforms (with keypoint-aware augmentation)
        # Note: albumentations preserves keypoint order, so surviving keypoints
        # maintain their original indices. This pattern matches CoNSeP/Ocelot datasets.
        if self.transforms:
            transformed = self.transforms(image=img, keypoints=detections)
            img = transformed["image"]
            detections = transformed["keypoints"]
            types = [types[idx] for idx, _ in enumerate(detections)]

        return img, detections, types, img_name

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, List[list], List[list], List[str]]:
        """Custom collate function for DataLoader.

        Stacks images into a batch tensor and keeps detections/types as lists
        (variable length per patch).

        Args:
            batch: List of (image, detections, types, name) tuples

        Returns:
            Tuple of:
                - images: (B, 3, H, W) tensor
                - detections: List of detection coordinate lists
                - types: List of type label lists
                - names: List of patch name strings
        """
        imgs, detections_list, types_list, names = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, list(detections_list), list(types_list), list(names)
