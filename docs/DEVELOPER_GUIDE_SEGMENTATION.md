# Developer Guide: Modifying inference_cellvit_experiment_segmentation.py

This comprehensive guide explains the architecture, components, and modification points of the `inference_cellvit_experiment_segmentation.py` script in detail.

---

## Table of Contents

1. [Script Overview](#script-overview)
2. [Architecture](#architecture)
3. [Code Structure](#code-structure)
4. [Key Components](#key-components)
5. [Detailed Method Documentation](#detailed-method-documentation)
6. [Execution Flow](#execution-flow)
7. [Modification Scenarios](#modification-scenarios)
8. [Customization Points](#customization-points)
9. [Extension Patterns](#extension-patterns)
10. [Data Structures](#data-structures)
11. [Debugging Tips](#debugging-tips)
12. [Common Pitfalls](#common-pitfalls)
13. [Best Practices](#best-practices)

---

## Script Overview

### Purpose

`inference_cellvit_experiment_segmentation.py` is a **generic nuclei segmentation evaluation script** that:
- Evaluates trained cell classifiers on segmentation datasets
- Calculates comprehensive metrics (overall + per-type)
- Works with flexible dataset structures
- Produces JSON output with all results

### Key Responsibilities

1. **Load Models:** CellViT (detection) + Classifier (classification)
2. **Load Dataset:** Using SegmentationDataset class
3. **Run Inference:** Extract cells and classify them
4. **Calculate Metrics:** Overall binary + per-type metrics
5. **Save Results:** JSON output with all metrics

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│  CellViTClassifierInferenceExperiment       │
│  (Parent Class)                              │
└──────────────────┬──────────────────────────┘
                   │ inherits
                   ▼
┌─────────────────────────────────────────────┐
│  SegmentationInferenceExperiment            │
│  (This Script)                               │
│                                              │
│  - Custom dataset loading                    │
│  - Segmentation-specific metrics             │
│  - Binary + per-type evaluation              │
└─────────────────────────────────────────────┘
```

---

## Architecture

### Class Hierarchy

```python
CellViTClassifierInferenceExperiment (Base)
    │
    ├── Provides: Model loading, basic inference loop
    ├── Provides: Logging, GPU management
    ├── Provides: Result aggregation
    │
    └── SegmentationInferenceExperiment (Child)
        │
        ├── Overrides: _load_dataset()
        ├── Overrides: _calculate_pipeline_scores()
        ├── Adds: Binary metric calculation
        └── Adds: Type-based metric calculation
```

### Component Relationships

```
┌──────────┐     ┌──────────────┐     ┌────────────────┐
│ CellViT  │────▶│ Cell         │────▶│ Classifier     │
│ Model    │     │ Extraction   │     │ Model          │
└──────────┘     └──────────────┘     └────────────────┘
      │                  │                      │
      │                  │                      │
      ▼                  ▼                      ▼
┌──────────────────────────────────────────────────────┐
│            Predictions (inst_map, type_map)           │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Metric Calculation   │
          │ - Binary metrics     │
          │ - Per-type metrics   │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ JSON Output          │
          └──────────────────────┘
```

---

## Code Structure

### File Organization

```python
# Lines 1-42: Header and imports setup
├── Shebang, encoding
├── Path manipulation
└── sys.path.append()

# Lines 43-114: Dependency checking
├── check_dependencies() function
└── Validates all required packages

# Lines 116-141: Imports
├── Standard library
├── Third-party packages
└── CellViT modules

# Lines 143-452: Main class
├── Class definition (line 143)
├── __init__() (lines 158-200)
├── _load_dataset() (lines 202-228)
├── _calculate_pipeline_scores() (lines 230-347)
├── run_inference() (lines 349-419)
├── parse_arguments() (lines 421-452)
└── Main execution (lines 454-460)
```

### Key Sections

**1. Dependency Checking (Lines 43-114)**
```python
def check_dependencies():
    """Validate required packages before importing"""
    # Check: cv2, torch, yaml, tqdm, sklearn, torchmetrics, pycm
    # If missing: Show clear error with pip install commands
```

**2. Class Definition (Line 143)**
```python
class SegmentationInferenceExperiment(CellViTClassifierInferenceExperiment):
    """Generic nuclei segmentation evaluation"""
```

**3. Initialization (Lines 158-200)**
```python
def __init__(self, ...):
    # Call parent constructor
    # Load models (CellViT + Classifier)
    # Setup logging
    # Load dataset
```

**4. Dataset Loading (Lines 202-228)**
```python
def _load_dataset(self) -> SegmentationDataset:
    # Load SegmentationDataset
    # Handle .npy or .mat files
    # Auto-detect folder structure
```

**5. Metric Calculation (Lines 230-347)**
```python
def _calculate_pipeline_scores(self, ...):
    # Calculate binary metrics (Dice, AJI, PQ, etc.)
    # Calculate per-type metrics
    # Aggregate results
```

**6. Main Execution (Lines 349-419)**
```python
def run_inference(self):
    # Loop through images
    # Extract cells with CellViT
    # Classify with classifier
    # Calculate metrics
    # Save results
```

---

## Key Components

### 1. SegmentationDataset

**Purpose:** Flexible dataset class for segmentation data

**Location:** `cellvit/training/datasets/segmentation_dataset.py`

**Features:**
- Supports .npy OR .mat files
- Configurable via label_map.yaml
- Auto-detects folder structure
- Returns: image, inst_map, type_map, image_name

**Usage in Script:**
```python
self.dataset = SegmentationDataset(
    dataset_path=self.dataset_path,
    split=self.split,
    # Loaded automatically
)
```

### 2. CellViT Model

**Purpose:** Detect and segment cells

**Type:** Pretrained vision transformer

**Output:** Cell embeddings + nuclei_type_map

**Usage:**
```python
# Loaded in parent __init__
self.cellvit_model = self._load_cellvit_model()

# Used during inference
predictions = self.cellvit_model(image_batch)
```

### 3. Classifier Model

**Purpose:** Classify detected cells into nuclei types

**Type:** Small MLP head trained on your data

**Output:** Class predictions for each cell

**Usage:**
```python
# Loaded in parent __init__
self.model = self._load_model()

# Used during inference (via postprocessor)
# Classifier is applied to cell embeddings
```

### 4. Postprocessor

**Purpose:** Extract cells from model predictions

**Type:** DetectionCellPostProcessorCupy

**Important:** Must use `cellvit_model.num_nuclei_classes`

**Usage:**
```python
cellvit_num_types = self.cellvit_model.num_nuclei_classes
postprocessor = DetectionCellPostProcessorCupy(
    wsi=None,
    nr_types=cellvit_num_types  # Critical!
)
```

### 5. Metric Calculators

**Binary Metrics:**
- `get_dice_1()` - Dice coefficient
- `get_fast_aji()` - Aggregated Jaccard Index
- `get_fast_aji_plus()` - AJI+ (strict)
- `get_fast_pq()` - Panoptic Quality

**Per-Type Metrics:**
- Type-specific Dice, AJI, PQ, DQ, SQ
- Cell-level F1, Precision, Recall

---

## Detailed Method Documentation

### `__init__(self, ...)`

**Purpose:** Initialize the experiment

**Parameters:**
- `logdir`: Training run directory
- `cellvit_path`: Path to CellViT model
- `dataset_path`: Path to dataset
- `outdir`: Output directory (optional)
- `checkpoint_name`: Classifier checkpoint (default: "model_best.pth")
- `split`: Dataset split (default: "test")
- `gpu`: GPU ID
- `magnification`: Magnification level (default: 40)

**Key Steps:**
1. Call parent `__init__`
2. Load CellViT model (`_load_cellvit_model()`)
3. Load classifier model (`_load_model()`)
4. Setup logging
5. Load dataset (`_load_dataset()`)

**Modification Point:**
- Override to add custom initialization
- Add custom attributes
- Load additional resources

**Example:**
```python
def __init__(self, ...):
    super().__init__(...)
    
    # Add custom initialization
    self.custom_param = custom_value
    self.load_additional_resources()
```

---

### `_load_dataset(self) -> SegmentationDataset`

**Purpose:** Load the segmentation dataset

**Returns:** `SegmentationDataset` instance

**Key Steps:**
1. Check dataset path exists
2. Initialize SegmentationDataset
3. Verify split folder exists
4. Log dataset info

**Modification Point:**
- Override to use different dataset class
- Add custom data validation
- Apply dataset preprocessing

**Example:**
```python
def _load_dataset(self) -> MyCustomDataset:
    """Load custom dataset"""
    dataset = MyCustomDataset(
        dataset_path=self.dataset_path,
        split=self.split,
        # Custom parameters
        custom_param=value
    )
    
    self.logger.info(f"Loaded custom dataset: {len(dataset)} images")
    return dataset
```

---

### `_calculate_pipeline_scores(self, pred_cells, gt_inst_map, gt_type_map, image_name)`

**Purpose:** Calculate all metrics for one image

**Parameters:**
- `pred_cells`: List of predicted cell dictionaries
- `gt_inst_map`: Ground truth instance map (H, W)
- `gt_type_map`: Ground truth type map (H, W)
- `image_name`: Name of the image

**Returns:** Dictionary with all metrics

**Key Steps:**
1. Convert predictions to instance/type maps
2. Calculate binary metrics (Dice, AJI, PQ)
3. Calculate per-type metrics
4. Aggregate results

**Structure:**
```python
def _calculate_pipeline_scores(self, ...):
    # Step 1: Create prediction maps
    pred_inst_map, pred_type_map = self.convert_cells_to_maps(pred_cells)
    
    # Step 2: Binary metrics
    binary_metrics = self.calculate_binary_metrics(
        pred_inst_map, gt_inst_map
    )
    
    # Step 3: Per-type metrics
    type_metrics = {}
    for class_id in self.nuclei_types:
        metrics = self.calculate_type_metrics(
            pred_type_map, gt_type_map, class_id
        )
        type_metrics[class_id] = metrics
    
    # Step 4: Aggregate
    return {
        'binary': binary_metrics,
        'per_type': type_metrics,
        'image_name': image_name
    }
```

**Modification Point:**
- Add new metrics here
- Custom metric calculation
- Filter or transform results

**Example: Adding Custom Metric**
```python
def _calculate_pipeline_scores(self, ...):
    # Existing code...
    binary_metrics = ...
    type_metrics = ...
    
    # Add custom metric
    custom_metric = self.calculate_custom_metric(
        pred_cells, gt_inst_map
    )
    
    return {
        'binary': binary_metrics,
        'per_type': type_metrics,
        'custom': custom_metric,  # Added!
        'image_name': image_name
    }

def calculate_custom_metric(self, pred_cells, gt_inst_map):
    """Your custom metric implementation"""
    # Calculate and return your metric
    return metric_value
```

---

### `run_inference(self)`

**Purpose:** Main inference loop

**Returns:** None (saves results to file)

**Key Steps:**
1. Create data loader
2. Loop through images
3. Extract cells with CellViT
4. Classify cells
5. Calculate metrics
6. Aggregate and save results

**Flow:**
```python
def run_inference(self):
    results = []
    
    for batch in tqdm(dataloader):
        # Step 1: Load data
        images, gt_inst_maps, gt_type_maps, image_names = batch
        
        # Step 2: Run CellViT
        with torch.no_grad():
            predictions = self.cellvit_model(images)
        
        # Step 3: Extract cells
        pred_cells = self.postprocess(predictions)
        
        # Step 4: Classify cells
        pred_cells = self.classify_cells(pred_cells)
        
        # Step 5: Calculate metrics
        scores = self._calculate_pipeline_scores(
            pred_cells, gt_inst_maps[0], gt_type_maps[0], image_names[0]
        )
        
        results.append(scores)
    
    # Step 6: Aggregate and save
    self.aggregate_and_save(results)
```

**Modification Point:**
- Add preprocessing steps
- Custom inference logic
- Save intermediate results
- Add visualization

**Example: Adding Visualization**
```python
def run_inference(self):
    # Existing inference code...
    
    for batch in tqdm(dataloader):
        # ... existing code ...
        
        # Add visualization
        if self.save_visualizations:
            self.visualize_predictions(
                images[0], pred_cells, 
                f"{self.outdir}/vis_{image_names[0]}.png"
            )
        
        results.append(scores)
    
    # ... rest of code ...

def visualize_predictions(self, image, cells, output_path):
    """Visualize predictions on image"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    for cell in cells:
        contour = cell['contour']
        ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
    
    plt.savefig(output_path)
    plt.close()
```

---

## Execution Flow

### Step-by-Step Execution

**1. Script Initialization**
```
├── Check dependencies
├── Import modules
└── Parse command-line arguments
```

**2. Experiment Setup**
```
├── Create SegmentationInferenceExperiment instance
├── Load CellViT model
├── Load classifier model
└── Load dataset
```

**3. Main Inference Loop**
```
For each image in dataset:
    ├── Load image and ground truth
    ├── Run CellViT inference
    │   └── Get cell detections + embeddings
    ├── Run classifier on embeddings
    │   └── Get cell type predictions
    ├── Calculate metrics
    │   ├── Binary: Dice, AJI, PQ, DQ, SQ
    │   └── Per-type: Dice, AJI, PQ, F1, Precision, Recall
    └── Store results
```

**4. Result Aggregation**
```
├── Aggregate all image results
├── Calculate average metrics
├── Create result dictionary
└── Save to JSON
```

**5. Output**
```
├── Console: Summary metrics
└── File: {logdir}/inference_results.json
```

---

## Modification Scenarios

### Scenario 1: Adding a New Metric

**Goal:** Add cell size distribution analysis

**Steps:**

**1. Add metric calculation method:**
```python
def calculate_size_distribution(self, cells):
    """Calculate cell size distribution statistics"""
    if not cells:
        return {'mean': 0, 'std': 0, 'count': 0}
    
    sizes = [cell.get('area', 0) for cell in cells]
    
    return {
        'mean': float(np.mean(sizes)),
        'std': float(np.std(sizes)),
        'min': float(np.min(sizes)),
        'max': float(np.max(sizes)),
        'count': len(sizes),
        'quartiles': {
            'q25': float(np.percentile(sizes, 25)),
            'q50': float(np.percentile(sizes, 50)),
            'q75': float(np.percentile(sizes, 75))
        }
    }
```

**2. Call in metric calculation:**
```python
def _calculate_pipeline_scores(self, pred_cells, gt_inst_map, gt_type_map, image_name):
    # Existing metrics...
    binary_metrics = self.calculate_binary_metrics(...)
    type_metrics = self.calculate_type_metrics(...)
    
    # Add size distribution
    size_dist = self.calculate_size_distribution(pred_cells)
    
    # Include in results
    results = {
        'binary_metrics': binary_metrics,
        'type_metrics': type_metrics,
        'size_distribution': size_dist,  # NEW!
        'image_name': image_name
    }
    
    return results
```

**3. Aggregate in final results:**
```python
def aggregate_results(self, all_results):
    # Existing aggregation...
    
    # Aggregate size distributions
    all_size_dists = [r['size_distribution'] for r in all_results]
    avg_size_dist = {
        'mean': np.mean([d['mean'] for d in all_size_dists]),
        'std': np.mean([d['std'] for d in all_size_dists]),
        # etc.
    }
    
    final_results['avg_size_distribution'] = avg_size_dist
    return final_results
```

---

### Scenario 2: Using a Custom Dataset Class

**Goal:** Use a custom dataset with different structure

**Steps:**

**1. Create custom dataset class:**
```python
# In cellvit/training/datasets/my_custom_dataset.py

from cellvit.training.datasets.base_dataset import BaseDataset

class MyCustomDataset(BaseDataset):
    def __init__(self, dataset_path, split, custom_param=None):
        super().__init__(dataset_path)
        self.split = split
        self.custom_param = custom_param
        
        # Custom initialization
        self.image_paths = self.find_images()
        self.label_paths = self.find_labels()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Custom loading logic
        image = self.load_image(self.image_paths[idx])
        inst_map, type_map = self.load_labels(self.label_paths[idx])
        image_name = self.image_paths[idx].stem
        
        return {
            'image': image,
            'inst_map': inst_map,
            'type_map': type_map,
            'image_name': image_name
        }
    
    def load_image(self, path):
        # Custom image loading
        pass
    
    def load_labels(self, path):
        # Custom label loading
        pass
```

**2. Override `_load_dataset()`:**
```python
def _load_dataset(self) -> MyCustomDataset:
    """Load custom dataset"""
    from cellvit.training.datasets.my_custom_dataset import MyCustomDataset
    
    dataset = MyCustomDataset(
        dataset_path=self.dataset_path,
        split=self.split,
        custom_param=self.custom_value
    )
    
    self.logger.info(f"Loaded custom dataset: {len(dataset)} images")
    return dataset
```

---

### Scenario 3: Adding Preprocessing

**Goal:** Apply custom preprocessing to images before inference

**Steps:**

**1. Add preprocessing method:**
```python
def preprocess_image(self, image):
    """Apply custom preprocessing"""
    # Example: Normalize
    image = image.astype(np.float32) / 255.0
    
    # Example: Apply CLAHE
    if len(image.shape) == 3:
        for i in range(3):
            image[:, :, i] = cv2.equalizeHist(
                (image[:, :, i] * 255).astype(np.uint8)
            ) / 255.0
    
    # Example: Resize
    if image.shape[0] != 256 or image.shape[1] != 256:
        image = cv2.resize(image, (256, 256))
    
    return image
```

**2. Apply in inference loop:**
```python
def run_inference(self):
    for batch in tqdm(dataloader):
        images, gt_inst_maps, gt_type_maps, image_names = batch
        
        # Apply preprocessing
        images = torch.stack([
            torch.from_numpy(self.preprocess_image(img.numpy()))
            for img in images
        ])
        
        # Continue with inference...
        predictions = self.cellvit_model(images)
        # ...
```

---

### Scenario 4: Saving Additional Outputs

**Goal:** Save visualizations and intermediate results

**Steps:**

**1. Add visualization method:**
```python
def save_prediction_visualization(self, image, pred_cells, gt_inst_map, output_path):
    """Create and save visualization"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(image)
    for cell in pred_cells:
        bbox = cell['bbox']
        rect = Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2]-bbox[0], bbox[3]-bbox[1],
            fill=False, edgecolor='r', linewidth=2
        )
        axes[1].add_patch(rect)
    axes[1].set_title(f'Predictions ({len(pred_cells)} cells)')
    axes[1].axis('off')
    
    # Ground truth
    axes[2].imshow(gt_inst_map, cmap='tab20')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**2. Call during inference:**
```python
def run_inference(self):
    # Create visualization directory
    vis_dir = os.path.join(self.outdir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for batch in tqdm(dataloader):
        # ... inference code ...
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"{image_names[0]}.png")
        self.save_prediction_visualization(
            images[0].cpu().numpy(),
            pred_cells,
            gt_inst_maps[0].cpu().numpy(),
            vis_path
        )
        
        # ... rest of code ...
```

---

### Scenario 5: Filtering Predictions

**Goal:** Filter out small or low-confidence cells

**Steps:**

**1. Add filtering method:**
```python
def filter_cells(self, cells, min_area=50, min_confidence=0.5):
    """Filter cells by size and confidence"""
    filtered = []
    
    for cell in cells:
        # Filter by area
        area = cell.get('area', 0)
        if area < min_area:
            continue
        
        # Filter by confidence
        confidence = cell.get('confidence', 1.0)
        if confidence < min_confidence:
            continue
        
        filtered.append(cell)
    
    self.logger.debug(
        f"Filtered {len(cells)} -> {len(filtered)} cells "
        f"(min_area={min_area}, min_conf={min_confidence})"
    )
    
    return filtered
```

**2. Apply after cell extraction:**
```python
def run_inference(self):
    for batch in tqdm(dataloader):
        # ... inference code ...
        pred_cells = self.postprocess(predictions)
        
        # Apply filtering
        pred_cells = self.filter_cells(
            pred_cells,
            min_area=self.min_cell_area,
            min_confidence=self.min_confidence
        )
        
        # Continue with filtered cells...
        scores = self._calculate_pipeline_scores(...)
```

---

## Customization Points

### 1. Command-Line Arguments

**Location:** `parse_arguments()` method

**Add Custom Argument:**
```python
@staticmethod
def parse_arguments():
    parser = argparse.ArgumentParser(...)
    
    # Existing arguments...
    parser.add_argument(...)
    
    # Add custom argument
    parser.add_argument(
        "--custom_param",
        type=str,
        default="default_value",
        help="Description of custom parameter"
    )
    
    return parser.parse_args()
```

**Use in __init__:**
```python
def __init__(self, custom_param=None, ...):
    super().__init__(...)
    self.custom_param = custom_param
```

---

### 2. Model Configuration

**Customize CellViT Loading:**
```python
def _load_cellvit_model(self):
    """Custom CellViT model loading"""
    model = super()._load_cellvit_model()
    
    # Apply custom modifications
    model.eval()
    model.freeze_backbone()  # Custom method
    
    return model
```

**Customize Classifier Loading:**
```python
def _load_model(self):
    """Custom classifier loading"""
    model = super()._load_model()
    
    # Apply modifications
    model.set_temperature(1.5)  # Example
    
    return model
```

---

### 3. Postprocessor Parameters

**Location:** `run_inference()` method

**Customize:**
```python
# In run_inference()
cellvit_num_types = self.cellvit_model.num_nuclei_classes
postprocessor = DetectionCellPostProcessorCupy(
    wsi=None,
    nr_types=cellvit_num_types,
    # Add custom parameters
    min_cell_size=100,  # Custom
    nms_threshold=0.5,  # Custom
    confidence_threshold=0.7  # Custom
)
```

---

### 4. Output Format

**Customize JSON Output:**
```python
def save_results(self, results):
    """Save results with custom format"""
    # Add metadata
    results['metadata'] = {
        'script_version': '1.0',
        'dataset': self.dataset_path,
        'custom_param': self.custom_param,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON
    output_path = os.path.join(self.logdir, 'inference_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save to CSV
    self.save_results_csv(results)
```

**Add CSV Export:**
```python
def save_results_csv(self, results):
    """Export results to CSV format"""
    import pandas as pd
    
    # Convert to DataFrame
    rows = []
    for image_result in results['image_results']:
        row = {
            'image_name': image_result['image_name'],
            'dice': image_result['binary_metrics']['dice'],
            'aji': image_result['binary_metrics']['aji'],
            # Add more metrics...
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save
    csv_path = os.path.join(self.logdir, 'inference_results.csv')
    df.to_csv(csv_path, index=False)
    self.logger.info(f"Saved CSV results to {csv_path}")
```

---

## Extension Patterns

### Pattern 1: Pre-Inference Hook

**Purpose:** Apply custom logic before inference on each batch

```python
def pre_inference_hook(self, images, gt_inst_maps, gt_type_maps, image_names):
    """Called before inference on each batch"""
    # Custom preprocessing
    images = self.custom_preprocess(images)
    
    # Custom validation
    self.validate_batch(images, gt_inst_maps, gt_type_maps)
    
    # Custom logging
    self.log_batch_info(images, image_names)
    
    return images, gt_inst_maps, gt_type_maps, image_names

def run_inference(self):
    for batch in tqdm(dataloader):
        images, gt_inst_maps, gt_type_maps, image_names = batch
        
        # Apply hook
        images, gt_inst_maps, gt_type_maps, image_names = \
            self.pre_inference_hook(images, gt_inst_maps, gt_type_maps, image_names)
        
        # Continue with inference...
```

---

### Pattern 2: Post-Inference Hook

**Purpose:** Modify predictions after inference

```python
def post_inference_hook(self, pred_cells, image, image_name):
    """Called after inference on each image"""
    # Filter predictions
    pred_cells = self.filter_cells(pred_cells)
    
    # Add custom attributes
    for cell in pred_cells:
        cell['custom_score'] = self.calculate_custom_score(cell)
    
    # Save intermediate results
    if self.save_intermediate:
        self.save_cells(pred_cells, image_name)
    
    return pred_cells

def run_inference(self):
    for batch in tqdm(dataloader):
        # ... inference code ...
        pred_cells = self.postprocess(predictions)
        
        # Apply hook
        pred_cells = self.post_inference_hook(
            pred_cells, images[0], image_names[0]
        )
        
        # Continue with metrics...
```

---

### Pattern 3: Custom Metric Aggregation

**Purpose:** Custom aggregation logic for metrics

```python
def aggregate_metrics(self, all_image_results):
    """Custom metric aggregation"""
    # Standard aggregation
    standard_agg = super().aggregate_metrics(all_image_results)
    
    # Add custom aggregations
    custom_agg = {
        'weighted_average': self.weighted_average(all_image_results),
        'per_class_breakdown': self.per_class_breakdown(all_image_results),
        'confidence_analysis': self.confidence_analysis(all_image_results)
    }
    
    # Combine
    return {**standard_agg, **custom_agg}

def weighted_average(self, results):
    """Calculate weighted average by image size"""
    total_pixels = sum(r['image_size'] for r in results)
    weighted_dice = sum(
        r['dice'] * r['image_size'] for r in results
    ) / total_pixels
    return {'weighted_dice': weighted_dice}
```

---

## Data Structures

### Cell Dictionary Structure

```python
cell = {
    # Geometry
    'bbox': [x_min, y_min, x_max, y_max],  # Bounding box
    'centroid': [x, y],  # Center point
    'contour': [[x1, y1], [x2, y2], ...],  # Contour points
    'area': float,  # Cell area in pixels
    
    # Classification
    'type': int,  # Nuclei type ID (1, 2, 3, etc.)
    'type_name': str,  # Nuclei type name
    'confidence': float,  # Classification confidence (0-1)
    
    # Features
    'token': np.ndarray,  # Embedding vector from CellViT
    'features': dict,  # Additional features
    
    # Instance
    'instance_id': int,  # Unique instance ID
    'mask': np.ndarray,  # Binary mask (optional)
}
```

### Results Dictionary Structure

```python
results = {
    # Overall binary metrics
    'overall_binary_metrics': {
        'dice': float,
        'aji': float,
        'aji_plus': float,
        'pq': float,
        'dq': float,
        'sq': float
    },
    
    # Per-type metrics
    'nuclei_metrics_d': {
        'Connective': {
            'dice_cell': float,
            'aji_cell': float,
            'pq_cell': float,
            'dq_cell': float,
            'sq_cell': float,
            'f1_cell': float,
            'prec_cell': float,
            'rec_cell': float,
            'num_pred': int,
            'num_gt': int
        },
        'Inflammatory': {...},
        'Neoplastic': {...}
    },
    
    # Per-image results
    'image_scores': [
        {
            'image_name': str,
            'binary_metrics': {...},
            'type_metrics': {...}
        },
        # ... more images
    ],
    
    # Metadata
    'metadata': {
        'dataset_path': str,
        'num_images': int,
        'num_classes': int,
        'class_names': list,
        'timestamp': str
    }
}
```

---

## Debugging Tips

### 1. Enable Verbose Logging

```python
# In __init__ or main
self.logger.setLevel(logging.DEBUG)

# Add debug statements
self.logger.debug(f"Processing image: {image_name}")
self.logger.debug(f"Predicted {len(pred_cells)} cells")
self.logger.debug(f"Metrics: {metrics}")
```

### 2. Save Intermediate Results

```python
def run_inference(self):
    # Create debug directory
    debug_dir = os.path.join(self.outdir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    for i, batch in enumerate(tqdm(dataloader)):
        # ... inference ...
        
        # Save predictions
        torch.save(
            predictions,
            os.path.join(debug_dir, f'predictions_{i}.pt')
        )
        
        # Save cells
        with open(os.path.join(debug_dir, f'cells_{i}.json'), 'w') as f:
            json.dump(pred_cells, f, indent=2, default=str)
```

### 3. Visualize Predictions

```python
def debug_visualize(self, image, pred_cells, gt_inst_map, output_path):
    """Create debug visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    
    # Predictions
    axes[0, 1].imshow(image)
    for cell in pred_cells:
        bbox = cell['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                        fill=False, edgecolor='r', linewidth=2)
        axes[0, 1].add_patch(rect)
        axes[0, 1].text(bbox[0], bbox[1], str(cell['type']), color='white')
    axes[0, 1].set_title(f'Predictions ({len(pred_cells)})')
    
    # Ground truth
    axes[1, 0].imshow(gt_inst_map, cmap='tab20')
    axes[1, 0].set_title('GT Instances')
    
    # Overlay
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(gt_inst_map, alpha=0.5, cmap='tab20')
    axes[1, 1].set_title('Overlay')
    
    plt.savefig(output_path)
    plt.close()
```

### 4. Check Data Statistics

```python
def analyze_dataset(self):
    """Analyze dataset statistics"""
    stats = {
        'num_images': len(self.dataset),
        'image_sizes': [],
        'num_cells_per_image': [],
        'type_distribution': {}
    }
    
    for i in range(len(self.dataset)):
        sample = self.dataset[i]
        stats['image_sizes'].append(sample['image'].shape)
        
        gt_type_map = sample['type_map']
        unique_types, counts = np.unique(gt_type_map[gt_type_map > 0], return_counts=True)
        
        for type_id, count in zip(unique_types, counts):
            if type_id not in stats['type_distribution']:
                stats['type_distribution'][type_id] = 0
            stats['type_distribution'][type_id] += count
    
    self.logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
```

---

## Common Pitfalls

### ⚠️ Pitfall 1: Wrong Number of Nuclei Types

**Problem:**
```python
# WRONG - using classifier's num_classes
postprocessor = DetectionCellPostProcessorCupy(
    wsi=None,
    nr_types=self.num_classes + 1  # ❌
)
```

**Solution:**
```python
# CORRECT - using CellViT's num_nuclei_classes
cellvit_num_types = self.cellvit_model.num_nuclei_classes
postprocessor = DetectionCellPostProcessorCupy(
    wsi=None,
    nr_types=cellvit_num_types  # ✅
)
```

---

### ⚠️ Pitfall 2: Not Moving Tensors to GPU

**Problem:**
```python
# Data on CPU, model on GPU → Error
predictions = self.cellvit_model(images)  # ❌
```

**Solution:**
```python
# Move to correct device
images = images.to(self.device)
predictions = self.cellvit_model(images)  # ✅
```

---

### ⚠️ Pitfall 3: Not Handling Empty Cell Lists

**Problem:**
```python
# Crashes if no cells detected
sizes = [cell['area'] for cell in pred_cells]
mean_size = np.mean(sizes)  # ❌ Fails if pred_cells is empty
```

**Solution:**
```python
# Handle empty case
if pred_cells:
    sizes = [cell['area'] for cell in pred_cells]
    mean_size = np.mean(sizes)
else:
    mean_size = 0  # ✅
```

---

### ⚠️ Pitfall 4: Breaking Inheritance

**Problem:**
```python
# Overriding without calling parent
def __init__(self, ...):
    # WRONG - doesn't call parent __init__
    self.my_custom_attr = value  # ❌
```

**Solution:**
```python
# Call parent first
def __init__(self, ...):
    super().__init__(...)  # ✅ Call parent
    self.my_custom_attr = value
```

---

### ⚠️ Pitfall 5: Modifying label_map.yaml Without Retraining

**Problem:**
- Change number of classes in label_map.yaml
- But classifier was trained with different num_classes
- Mismatch causes errors

**Solution:**
- Ensure label_map.yaml matches training configuration
- Or retrain classifier if changing classes

---

## Best Practices

### ✅ 1. Keep Inheritance Clean

```python
# Good: Minimal override, call parent when needed
def _load_dataset(self):
    # Add custom validation
    if not self.dataset_path.exists():
        raise ValueError(f"Dataset not found: {self.dataset_path}")
    
    # Call parent or provide custom implementation
    return super()._load_dataset()
```

### ✅ 2. Use Logging Effectively

```python
# Good: Informative logging at different levels
self.logger.info(f"Starting inference on {len(self.dataset)} images")
self.logger.debug(f"Processing image {i}/{len(self.dataset)}")
self.logger.warning(f"No cells detected in image {image_name}")
self.logger.error(f"Failed to process {image_name}: {str(e)}")
```

### ✅ 3. Validate Inputs

```python
# Good: Validate before processing
def _calculate_pipeline_scores(self, pred_cells, gt_inst_map, gt_type_map, image_name):
    # Validate inputs
    assert gt_inst_map.shape == gt_type_map.shape, "Shape mismatch"
    assert gt_inst_map.ndim == 2, "Expected 2D array"
    assert isinstance(pred_cells, list), "pred_cells must be list"
    
    # Continue with calculation
    ...
```

### ✅ 4. Handle Edge Cases

```python
# Good: Handle edge cases gracefully
def calculate_metrics(self, pred_cells, gt_inst_map):
    # Handle empty predictions
    if not pred_cells:
        return self.get_default_metrics()
    
    # Handle empty ground truth
    if gt_inst_map.max() == 0:
        return self.get_default_metrics()
    
    # Normal calculation
    return self._calculate_metrics_impl(pred_cells, gt_inst_map)
```

### ✅ 5. Test Incrementally

```bash
# Test with single image first
python3 inference_script.py --dataset_path ./test_single_image ...

# Then test with small batch
python3 inference_script.py --dataset_path ./test_small_batch ...

# Finally full dataset
python3 inference_script.py --dataset_path ./full_dataset ...
```

### ✅ 6. Document Your Changes

```python
# Good: Clear documentation
def custom_metric(self, pred_cells, gt_cells):
    """
    Calculate custom overlap metric.
    
    This metric measures the percentage of predicted cells that
    overlap with ground truth cells by at least 50%.
    
    Args:
        pred_cells: List of predicted cell dictionaries
        gt_cells: List of ground truth cell dictionaries
    
    Returns:
        float: Overlap percentage (0-100)
    
    Note:
        Returns 0 if pred_cells is empty.
    """
    if not pred_cells:
        return 0.0
    
    # Implementation...
```

### ✅ 7. Follow Code Style

```python
# Good: Consistent with existing code
# - Use snake_case for methods
# - Use descriptive variable names
# - Add type hints
# - Keep methods focused

def calculate_per_type_metrics(
    self, 
    pred_type_map: np.ndarray, 
    gt_type_map: np.ndarray,
    class_id: int
) -> Dict[str, float]:
    """Calculate metrics for specific nuclei type"""
    # Implementation
    pass
```

---

## See Also

### Related Scripts
- **Base class:** `inference_cellvit_experiment_classifier.py`
- **CRC CODEX version:** `inference_cellvit_experiment_crccodex.py`
- **CoNSeP version:** `inference_cellvit_experiment_consep.py`
- **PanNuke version:** `inference_cellvit_experiment_pannuke.py`

### Related Modules
- **Dataset:** `cellvit/training/datasets/segmentation_dataset.py`
- **Postprocessor:** `cellvit/training/evaluate/cellpostprocessor.py`
- **Metrics:** `cellvit/evaluation/metrics.py`
- **Models:** `cellvit/models/`

### Documentation
- **User Guide:** `UNDERSTANDING_SEGMENTATION_SCRIPTS.md`
- **Comparison:** `COMPARISON_SEGMENTATION_SCRIPTS.md`
- **Quick Start:** `SEGMENTATION_EVALUATION_QUICKSTART.md`
- **How-To:** `HOW_TO_RUN_SEGMENTATION_EVALUATION.md`

---

## Summary

This developer guide covers:

✅ **Architecture** - How the script is structured
✅ **Components** - Key classes and methods
✅ **Flow** - Step-by-step execution
✅ **Methods** - Detailed documentation of each method
✅ **Scenarios** - Real modification examples
✅ **Patterns** - Extension patterns and hooks
✅ **Data Structures** - Format of inputs and outputs
✅ **Debugging** - Tips for troubleshooting
✅ **Pitfalls** - Common mistakes to avoid
✅ **Best Practices** - Professional coding standards

With this guide, you can confidently modify `inference_cellvit_experiment_segmentation.py` for your specific needs!
