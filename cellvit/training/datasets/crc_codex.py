# -*- coding: utf-8 -*-
# CRC CODEX Dataset
#
# Custom dataset for CRC CODEX data with .npy format
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
from pathlib import Path
from typing import Callable, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchstain
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import tqdm
import csv
from natsort import natsorted as sorted


class CRCCodexDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        filelist_path: Union[Path, str] = None,
        transforms: Callable = A.Compose(
            [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        ),
        normalize_stains: bool = False,
    ) -> None:
        """CRC CODEX Dataset for Cell Segmentation

        Dataset with .npy files containing inst_map and type_map as dictionaries.
        Three nuclei types: 1 (Connective), 2 (Inflammatory), 3 (Neoplastic)

        Args:
            dataset_path (Union[Path, str]): Path to the dataset parent folder
            split (str): Split of the dataset (train, test, val, etc.)
            filelist_path (Union[Path, str], optional): Path to a filelist (csv) to retrieve just a subset of images to use.
                Otherwise, all images from split are used. Defaults to None.
            transforms (Callable, optional): Transformations. Defaults to A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]).
            normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        """
        super().__init__()
        self.transforms = transforms
        self.normalize_stains = normalize_stains
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split / "images"
        self.label_path = self.dataset_path / self.split / "labels"

        # Get all .npy image files
        self.images = [f for f in sorted(self.image_path.glob("*.npy"))]
        if filelist_path is not None:
            selected_files = []
            with open(filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    selected_files.append(row[0])
            self.images = [f for f in self.images if f.stem in selected_files]

        # Get corresponding label files
        self.labels = []
        for img_path in self.images:
            img_name = img_path.stem
            self.labels.append(self.label_path / f"{img_name}.npy")

        self.cache_images = {}
        self.cache_labels = {}
        
        # CRC CODEX nuclei types
        # 0: Background
        # 1: Connective
        # 2: Inflammatory
        # 3: Neoplastic
        self.type_nuclei_dict = {
            0: "Background",
            1: "Connective",
            2: "Inflammatory",
            3: "Neoplastic",
        }

    def cache_dataset(self) -> None:
        """Cache the dataset in memory"""
        for img_path, label_path in tqdm.tqdm(
            zip(self.images, self.labels),
            desc="Caching dataset",
            total=len(self.images),
        ):
            # Load image
            img = np.load(img_path)
            if img.dtype == np.float64 or img.dtype == np.float32:
                # If normalized, convert back to uint8
                img = (img * 255).astype(np.uint8)
            if len(img.shape) == 2:
                # If grayscale, convert to RGB
                img = np.stack([img, img, img], axis=-1)
            self.cache_images[img_path.stem] = img

            # Load label
            label = np.load(label_path, allow_pickle=True)
            self.cache_labels[label_path.stem] = label

    def __len__(self) -> int:
        """Get length of dataset

        Returns:
            int: Length of dataset
        """
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        """Get one dataset item

        Args:
            index (int): Index

        Returns:
            dict: Dataset entry with keys:
                * image: Image as torch.Tensor with shape (3, H, W)
                * inst_map: Instance map as numpy array with shape (H, W)
                * type_map: Type map as numpy array with shape (H, W)
                * image_name: Image name (stem)
        """
        img_path = self.images[index]
        label_path = self.labels[index]

        # Load from cache or disk
        if img_path.stem in self.cache_images:
            img = self.cache_images[img_path.stem].copy()
            label_dict = self.cache_labels[label_path.stem]
        else:
            # Load image
            img = np.load(img_path)
            if img.dtype == np.float64 or img.dtype == np.float32:
                # If normalized, convert back to uint8
                img = (img * 255).astype(np.uint8)
            if len(img.shape) == 2:
                # If grayscale, convert to RGB
                img = np.stack([img, img, img], axis=-1)

            # Load label
            label_dict = np.load(label_path, allow_pickle=True)

        # Extract inst_map and type_map from the dictionary
        if isinstance(label_dict, np.ndarray) and label_dict.shape == ():
            # If it's a 0-d array containing a dict
            label_dict = label_dict.item()
        
        inst_map = label_dict.get("inst_map")
        type_map = label_dict.get("type_map")

        # Apply stain normalization if requested
        if self.normalize_stains:
            img, _, _ = self.normalizer.normalize(I=img, stains=False)

        # Apply transformations
        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        return {
            "image": img,
            "inst_map": inst_map,
            "type_map": type_map,
            "image_name": img_path.stem,
        }

    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by number of cells
        
        Args:
            gamma (float, optional): Gamma scaling factor. Defaults to 1.
            
        Returns:
            torch.Tensor: Sampling weights
        """
        weights = []
        for label_path in self.labels:
            label_dict = np.load(label_path, allow_pickle=True)
            if isinstance(label_dict, np.ndarray) and label_dict.shape == ():
                label_dict = label_dict.item()
            inst_map = label_dict.get("inst_map")
            num_cells = len(np.unique(inst_map)) - 1  # -1 for background
            weights.append(num_cells)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        weights = np.power(weights, gamma)
        weights = weights / weights.sum()
        
        return torch.from_numpy(weights).float()

    def get_nuclei_types(self) -> dict:
        """Get nuclei types dictionary

        Returns:
            dict: Dictionary mapping type indices to type names
        """
        return self.type_nuclei_dict
