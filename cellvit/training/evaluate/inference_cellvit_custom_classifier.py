# -*- coding: utf-8 -*-
# Simplified CellViT Evaluation Script for Custom Classifiers
# 
# This script provides an easy-to-use interface for evaluating custom classifiers
# trained with train_cell_classifier_head.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

"""
Simplified evaluation script for custom CellViT classifiers.

This script wraps the existing detection evaluation functionality with a more
user-friendly interface and better error messages.

Usage:
    python3 inference_cellvit_custom_classifier.py \
        --logdir /path/to/trained/classifier \
        --dataset_path /path/to/dataset \
        --cellvit_path /path/to/cellvit/model.pth \
        --input_shape 256 256

For detailed documentation, see docs/EVALUATION_GUIDE.md
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from cellvit.training.evaluate.inference_cellvit_experiment_detection import (
    CellViTInfExpDetection,
)


class CustomClassifierEvaluator:
    """Simplified evaluator for custom CellViT classifiers"""

    def __init__(
        self,
        logdir: str,
        dataset_path: str,
        cellvit_path: str,
        input_shape: list,
        normalize_stains: bool = False,
        gpu: int = 0,
    ):
        """
        Initialize the evaluator.

        Args:
            logdir: Path to the directory containing your trained classifier
            dataset_path: Path to your dataset parent directory
            cellvit_path: Path to the CellViT base model (.pth file)
            input_shape: List of [height, width] for input images
            normalize_stains: Whether to apply stain normalization
            gpu: GPU ID to use for inference
        """
        self.logdir = Path(logdir)
        self.dataset_path = Path(dataset_path)
        self.cellvit_path = Path(cellvit_path)
        self.input_shape = input_shape
        self.normalize_stains = normalize_stains
        self.gpu = gpu

    def validate_inputs(self) -> bool:
        """
        Validate that all required files and directories exist.

        Returns:
            bool: True if validation passes

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If configuration is invalid
        """
        print("=" * 80)
        print("VALIDATING INPUTS")
        print("=" * 80)

        # Check logdir
        if not self.logdir.exists():
            raise FileNotFoundError(
                f"Training log directory not found: {self.logdir}\n"
                f"Please provide the path to the directory created by train_cell_classifier_head.py"
            )
        print(f"✓ Log directory found: {self.logdir}")

        # Check for config.yaml
        config_path = self.logdir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"The log directory should contain a config.yaml file.\n"
                f"Did you provide the correct training output directory?"
            )
        print(f"✓ Configuration file found: {config_path}")

        # Check for checkpoint
        checkpoint_dir = self.logdir / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoints directory not found: {checkpoint_dir}\n"
                f"The log directory should contain a checkpoints/ folder with model_best.pth"
            )

        checkpoint_path = checkpoint_dir / "model_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best model checkpoint not found: {checkpoint_path}\n"
                f"Please ensure training completed successfully and the checkpoint was saved."
            )
        print(f"✓ Model checkpoint found: {checkpoint_path}")

        # Check dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}\n"
                f"Please provide the path to your dataset parent directory."
            )
        print(f"✓ Dataset directory found: {self.dataset_path}")

        # Check for test data
        test_dir = self.dataset_path / "test"
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}\n"
                f"Your dataset should have a 'test' folder with 'images' and 'labels' subdirectories."
            )

        test_images = test_dir / "images"
        test_labels = test_dir / "labels"
        if not test_images.exists():
            raise FileNotFoundError(
                f"Test images directory not found: {test_images}\n"
                f"Please ensure your test dataset has the correct structure."
            )
        if not test_labels.exists():
            raise FileNotFoundError(
                f"Test labels directory not found: {test_labels}\n"
                f"Please ensure your test dataset has the correct structure."
            )
        print(f"✓ Test data found: {test_dir}")

        # Check CellViT model
        if not self.cellvit_path.exists():
            raise FileNotFoundError(
                f"CellViT model not found: {self.cellvit_path}\n"
                f"Please download the CellViT model checkpoint.\n"
                f"See: https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79"
            )
        print(f"✓ CellViT model found: {self.cellvit_path}")

        # Validate input shape
        if len(self.input_shape) != 2:
            raise ValueError(
                f"Input shape must be exactly 2 values [height, width], got {len(self.input_shape)}"
            )
        if self.input_shape[0] <= 0 or self.input_shape[1] <= 0:
            raise ValueError(
                f"Input shape dimensions must be positive, got {self.input_shape}"
            )
        print(f"✓ Input shape: {self.input_shape[0]} x {self.input_shape[1]}")

        print("=" * 80)
        print("VALIDATION PASSED - Ready to run evaluation")
        print("=" * 80)
        print()

        return True

    def print_configuration(self):
        """Print the evaluation configuration"""
        print("=" * 80)
        print("EVALUATION CONFIGURATION")
        print("=" * 80)
        print(f"Log Directory:        {self.logdir}")
        print(f"Dataset:              {self.dataset_path}")
        print(f"CellViT Model:        {self.cellvit_path}")
        print(f"Input Shape:          {self.input_shape[0]} x {self.input_shape[1]}")
        print(f"Stain Normalization:  {self.normalize_stains}")
        print(f"GPU:                  {self.gpu}")
        print("=" * 80)
        print()

    def run_evaluation(self):
        """Run the evaluation"""
        # Validate inputs
        self.validate_inputs()

        # Print configuration
        self.print_configuration()

        # Create experiment
        print("Initializing evaluation experiment...")
        experiment = CellViTInfExpDetection(
            logdir=str(self.logdir),
            cellvit_path=str(self.cellvit_path),
            dataset_path=str(self.dataset_path),
            normalize_stains=self.normalize_stains,
            gpu=self.gpu,
            input_shape=self.input_shape,
        )

        # Run inference
        print("Starting inference on test set...")
        print("This may take several minutes depending on dataset size...")
        print()
        experiment.run_inference()

        # Print completion message
        print()
        print("=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        results_dir = self.logdir / "inference_results"
        print(f"Results saved to: {results_dir}")
        print()
        print("Generated files:")
        print(f"  - {results_dir / 'inference_results.json'}")
        print(f"  - {results_dir / 'confusion_matrix.png'}")
        print(f"  - {results_dir / 'confusion_matrix_normalized.png'}")
        print(f"  - {results_dir / 'inference.log'}")
        print()
        print("Review the JSON file for detailed metrics.")
        print("Check the confusion matrices to understand classification performance.")
        print("=" * 80)


def create_parser():
    """Create argument parser with helpful descriptions"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
╔════════════════════════════════════════════════════════════════════════════╗
║          CellViT++ Custom Classifier Evaluation Script                     ║
╚════════════════════════════════════════════════════════════════════════════╝

This script evaluates custom classifiers trained with train_cell_classifier_head.py

REQUIREMENTS:
  1. Trained classifier directory (output from train_cell_classifier_head.py)
  2. Dataset in the same format used for training
  3. CellViT base model checkpoint used during training
  4. Input image dimensions used during training

EXAMPLE USAGE:
  python3 ./cellvit/training/evaluate/inference_cellvit_custom_classifier.py \\
    --logdir ./logs_local/CellViT-Classifier_2024_01_15_120000 \\
    --dataset_path ./test_database/training_database/Example-Detection \\
    --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \\
    --input_shape 256 256

For detailed documentation, see: docs/EVALUATION_GUIDE.md
        """,
    )

    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Path to the trained classifier directory (contains checkpoints/ and config.yaml)",
        metavar="PATH",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset parent directory (contains train/, test/, splits/)",
        metavar="PATH",
    )

    parser.add_argument(
        "--cellvit_path",
        type=str,
        required=True,
        help="Path to the CellViT base model checkpoint (.pth file)",
        metavar="PATH",
    )

    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=2,
        required=True,
        help="Input image shape as two integers: height width (e.g., 256 256)",
        metavar=("HEIGHT", "WIDTH"),
    )

    parser.add_argument(
        "--normalize_stains",
        action="store_true",
        help="Apply stain normalization (use if enabled during training)",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use for inference (default: 0)",
        metavar="ID",
    )

    return parser


def main():
    """Main entry point"""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Print header
    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║          CellViT++ Custom Classifier Evaluation                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Create evaluator
        evaluator = CustomClassifierEvaluator(
            logdir=args.logdir,
            dataset_path=args.dataset_path,
            cellvit_path=args.cellvit_path,
            input_shape=args.input_shape,
            normalize_stains=args.normalize_stains,
            gpu=args.gpu,
        )

        # Run evaluation
        evaluator.run_evaluation()

    except (FileNotFoundError, ValueError) as e:
        print()
        print("=" * 80)
        print("ERROR: Validation failed")
        print("=" * 80)
        print(str(e))
        print("=" * 80)
        print()
        print("Need help? Check the documentation:")
        print("  docs/EVALUATION_GUIDE.md")
        print()
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR: Evaluation failed")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        print("If this error persists, please:")
        print("  1. Check that all paths are correct")
        print("  2. Verify your dataset structure matches training")
        print("  3. Ensure you have enough GPU memory")
        print("  4. Review docs/EVALUATION_GUIDE.md for troubleshooting")
        print("=" * 80)
        print()
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
