# Guide: Replacing Foundation Models in CellViT++

This guide explains how the foundation model infrastructure works in CellViT++ and provides step-by-step instructions for integrating additional foundation models.

## Table of Contents
1. [Understanding the Model Architecture](#understanding-the-model-architecture)
2. [Foundation Model Infrastructure](#foundation-model-infrastructure)
3. [Available Foundation Models](#available-foundation-models)
4. [How to Add a New Foundation Model](#how-to-add-a-new-foundation-model)
5. [Step-by-Step Implementation Example](#step-by-step-implementation-example)
6. [Model Loading and Inference](#model-loading-and-inference)
7. [Troubleshooting](#troubleshooting)

---

## Understanding the Model Architecture

CellViT++ uses a U-Net-like architecture with Vision Transformer (ViT) backbones as encoders. The model consists of:

1. **Encoder (Foundation Model)**: A pretrained Vision Transformer that extracts features from input images
2. **Decoder**: Multiple upsampling branches for different tasks:
   - Binary nuclei segmentation
   - Horizontal-Vertical (HV) map for instance separation
   - Nuclei type classification
3. **Skip Connections**: Connect encoder layers to decoder for better spatial information

### Key Components

```
Input Image (1024x1024 or 256x256)
    ↓
Foundation Model Encoder (SAM/Virchow/UNI/HIPT/etc.)
    ↓ (4 skip connections from layers)
    z1, z2, z3, z4
    ↓
Shared Decoder Layers
    ↓
Branch Decoders (nuclei_binary_map, hv_map, nuclei_type_map)
    ↓
Output Predictions
```

---

## Foundation Model Infrastructure

### Directory Structure

The foundation model implementation follows this structure:

```
cellvit/models/
├── cell_segmentation/
│   ├── cellvit.py              # Base CellViT model
│   ├── cellvit_256.py          # CellViT with HIPT-256 backbone
│   ├── cellvit_sam.py          # CellViT with SAM backbone (B/L/H variants)
│   ├── cellvit_uni.py          # CellViT with UNI backbone
│   ├── cellvit_virchow.py      # CellViT with Virchow backbone
│   ├── cellvit_virchow2.py     # CellViT with Virchow2 backbone
│   └── backbones.py            # Backbone ViT implementations
├── utils/
│   ├── sam_utils.py            # SAM-specific utilities
│   ├── uni_utils.py            # UNI-specific utilities
│   └── virchow_utils.py        # Virchow-specific utilities
└── base/
    └── vision_transformer.py   # Base ViT implementation
```

### Core Model Classes

Each foundation model integration creates a new class that inherits from the base `CellViT` class:

- **CellViT**: Base model with standard ViT backbone
- **CellViT256**: Uses HIPT-256 pretrained model
- **CellViTSAM**: Uses Segment Anything Model (SAM-B/L/H)
- **CellViTUNI**: Uses UNI foundation model
- **CellViTVirchow**: Uses Virchow foundation model
- **CellViTVirchow2**: Uses Virchow2 foundation model with register tokens

---

## Available Foundation Models

### 1. SAM (Segment Anything Model)
- **Variants**: SAM-B (Base), SAM-L (Large), SAM-H (Huge)
- **Embed Dim**: 768 (B), 1024 (L), 1280 (H)
- **Depth**: 12 (B), 24 (L), 32 (H)
- **Patch Size**: 16
- **Source**: Meta AI
- **File**: `cellvit_sam.py`

### 2. UNI (Universal)
- **Embed Dim**: 1024
- **Depth**: 24
- **Patch Size**: 16
- **Source**: Mahmood Lab (Nature Medicine 2024)
- **Links**: [GitHub](https://github.com/mahmoodlab/UNI) | [Paper](https://www.nature.com/articles/s41591-024-02857-3)
- **File**: `cellvit_uni.py`

### 3. Virchow
- **Embed Dim**: 1280
- **Depth**: 32
- **Patch Size**: 14
- **MLP Ratio**: 5.3375 (uses SwiGLU)
- **Source**: Paige AI (Nature Medicine 2024)
- **Links**: [HuggingFace](https://huggingface.co/paige-ai/Virchow) | [Paper](https://doi.org/10.1038/s41591-024-03141-0)
- **File**: `cellvit_virchow.py`

### 4. Virchow2
- **Embed Dim**: 1280
- **Depth**: 32
- **Patch Size**: 14
- **MLP Ratio**: 5.3375 (uses SwiGLU)
- **Register Tokens**: 4
- **Source**: Paige AI (arXiv 2024)
- **Links**: [HuggingFace](https://huggingface.co/paige-ai/Virchow2) | [Paper](https://doi.org/10.48550/arXiv.2408.00738)
- **File**: `cellvit_virchow2.py`

### 5. HIPT-256
- **Embed Dim**: 384
- **Depth**: 12
- **Patch Size**: 16
- **Source**: Mahmood Lab
- **File**: `cellvit_256.py`

---

## How to Add a New Foundation Model

Adding a new foundation model requires creating three main components:

### 1. Backbone Implementation (in `backbones.py`)

Create a new backbone class that:
- Inherits from the appropriate base ViT class
- Implements the `forward` method to return: `(classifier_output, cls_token, extracted_layers)`
- Extracts features from specified layers for skip connections

### 2. Model Implementation (new file: `cellvit_<your_model>.py`)

Create a new CellViT variant that:
- Inherits from `CellViT` base class
- Initializes model parameters (embed_dim, depth, num_heads, etc.)
- Creates the backbone encoder
- Loads pretrained weights
- Optionally overrides `forward` method if needed

### 3. Utilities (optional: `utils/<your_model>_utils.py`)

If your foundation model requires special components:
- Custom layers (e.g., SwiGLU for Virchow)
- Custom ViT implementations
- Helper functions

### 4. Inference Integration (update `inference/inference_disk.py`)

Update the `_get_model` method to include your new model type.

---

## Step-by-Step Implementation Example

Let's walk through adding a hypothetical new foundation model called "SuperViT".

### Step 1: Define Model Parameters

First, understand your foundation model's architecture:
```python
# SuperViT parameters
embed_dim = 1536        # Embedding dimension
depth = 40              # Number of transformer blocks
num_heads = 24          # Number of attention heads
patch_size = 14         # Patch size
img_size = 224          # Input image size
extract_layers = [10, 20, 30, 40]  # Layers for skip connections (4 required)
```

### Step 2: Create Backbone Class

Add to `cellvit/models/cell_segmentation/backbones.py`:

```python
from cellvit.models.utils.supervit_utils import SuperViTTransformer  # If needed

class ViTCellViTSuperViT(SuperViTTransformer):
    """Backbone for SuperViT foundation model
    
    Args:
        extract_layers (List[int]): Transformer block indices for skip connections
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Patch size. Defaults to 14.
        depth (int): Number of transformer blocks. Defaults to 40.
        num_heads (int): Number of attention heads. Defaults to 24.
        embed_dim (int): Embedding dimension. Defaults to 1536.
        num_classes (int): Number of output classes. Defaults to 0.
    """
    
    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 224,
        patch_size: int = 14,
        depth: int = 40,
        num_heads: int = 24,
        embed_dim: int = 1536,
        num_classes: int = 0,
        **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_classes=0,  # We handle classification separately
            **kwargs
        )
        self.extract_layers = extract_layers
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass with skip connections
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            Tuple containing:
                - classifier_output: Classification output (B, num_classes)
                - cls_token: Class token features (B, embed_dim)
                - extracted_layers: List of 4 feature maps for skip connections
        """
        extracted_layers = []
        
        # Patch embedding
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        # Forward through transformer blocks
        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)
        
        # Classification head
        output = self.head(x[:, 0])
        
        return output, x[:, 0], extracted_layers
```

### Step 3: Create CellViT Variant

Create `cellvit/models/cell_segmentation/cellvit_supervit.py`:

```python
# -*- coding: utf-8 -*-
#
# CellViT-SuperViT model
#
# @ Your Name, your.email@institution.edu
# Institution

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.backbones import ViTCellViTSuperViT
from cellvit.models.utils.blocks import Conv2DBlock, Deconv2DBlock


class CellViTSuperViT(CellViT):
    """CellViT with SuperViT backbone settings
    
    SuperViT Links:
        Paper: https://doi.org/10.xxxx/your-paper
        HuggingFace: https://huggingface.co/your-org/SuperViT
    
    Args:
        model_supervit_path (Union[Path, str]): Path to SuperViT checkpoint
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connections. Defaults to 0.
    """
    
    def __init__(
        self,
        model_supervit_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        # Call parent constructor to avoid re-initializing
        super(CellViT, self).__init__()
        
        # Set SuperViT-specific parameters
        self.img_size = 224
        self.patch_size = 14
        self.embed_dim = 1536
        self.depth = 40
        self.num_heads = 24
        self.qkv_bias = True
        self.extract_layers = [10, 20, 30, 40]
        self.input_channels = 3
        self.mlp_ratio = 4.0
        self.drop_rate = drop_rate
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.model_supervit_path = model_supervit_path
        
        # For models with different patch sizes, you may need rescaling
        # e.g., Virchow uses patch_size=14 and needs rescaling
        self.input_rescale_dict = {256: 252, 1024: 1022}
        
        regression_loss = False
        
        # Create encoder backbone
        self.encoder = ViTCellViTSuperViT(
            extract_layers=self.extract_layers,
            num_classes=num_tissue_classes,
            img_size=self.img_size,
            patch_size=self.patch_size,
            depth=self.depth,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
        )
        
        # Setup skip connection dimensions based on embed_dim
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        
        # Create decoder layers (shared across branches)
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )
        
        # Setup branch outputs
        self.regression_loss = regression_loss
        offset_branches = 2 if self.regression_loss else 0
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }
        
        # Create branch decoders
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            2 + offset_branches
        )
        self.hv_map_decoder = self.create_upsampling_branch(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )
        
        # Load pretrained weights
        self.load_pretrained_encoder(self.model_supervit_path)
    
    def load_pretrained_encoder(self, model_supervit_path: Union[Path, str]):
        """Load pretrained SuperViT encoder from provided path
        
        Args:
            model_supervit_path (Union[Path, str]): Path to SuperViT checkpoint
        """
        if model_supervit_path is None:
            print("No checkpoint provided!")
        else:
            state_dict = torch.load(str(model_supervit_path), map_location="cpu")
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loading checkpoint: {msg}")
    
    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass
        
        Args:
            x (torch.Tensor): Input images in BCHW format
            retrieve_tokens (bool, optional): Return tokens. Defaults to False.
        
        Returns:
            dict: Predictions for all branches:
                - tissue_types: Tissue classification (B, num_tissue_classes)
                - nuclei_binary_map: Binary segmentation (B, 2, H, W)
                - hv_map: HV map (B, 2, H, W)
                - nuclei_type_map: Nuclei type prediction (B, num_nuclei_classes, H, W)
                - [Optional] tokens: Feature tokens (B, embed_dim, H', W')
                - [Optional] regression_map: Regression map (B, 2, H, W)
        """
        out_dict = {}
        input_shape = x.shape[2]
        
        # Optionally rescale input if needed (for patch_size != 16)
        if hasattr(self, 'input_rescale_dict') and input_shape in self.input_rescale_dict:
            rescale_value = self.input_rescale_dict[input_shape]
            x = F.interpolate(x, size=(rescale_value, rescale_value), mode="area")
        else:
            rescale_value = input_shape
        
        # Forward through encoder
        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits
        
        # Unpack skip connections
        z0, z1, z2, z3, z4 = x if input_shape != rescale_value else x, *z
        
        # Reshape features for decoder (restore spatial dimensions)
        patch_dim = [int(d / self.patch_size) for d in [rescale_value, rescale_value]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        
        # Forward through decoders
        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder, input_shape
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder, input_shape
            )
        
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder, input_shape
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder, input_shape
        )
        
        if retrieve_tokens:
            out_dict["tokens"] = z4
        
        return out_dict
    
    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
        rescale_value: int = None,
    ) -> torch.Tensor:
        """Forward upsample branch
        
        Args:
            z0-z4: Skip connection features
            branch_decoder: Decoder network for specific branch
            rescale_value: Target size for final output
        
        Returns:
            torch.Tensor: Upsampled predictions
        """
        # Use default implementation or override if needed
        # For models with rescaling (patch_size != 16):
        if rescale_value is not None and hasattr(self, 'input_rescale_dict'):
            b4 = branch_decoder.bottleneck_upsampler(z4)
            b3 = self.decoder3(z3)
            b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
            b2 = self.decoder2(z2)
            b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
            b1 = self.decoder1(z1)
            b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
            b1 = F.interpolate(
                b1, size=(rescale_value, rescale_value), 
                mode="bilinear", align_corners=False
            )
            b0 = self.decoder0(z0)
            b0 = F.interpolate(
                b0, size=(rescale_value, rescale_value),
                mode="bilinear", align_corners=False
            )
            branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
            return branch_output
        else:
            # Use default implementation from parent class
            return super()._forward_upsample(z0, z1, z2, z3, z4, branch_decoder)
```

### Step 4: Update Inference Code

Update `cellvit/inference/inference_disk.py`:

```python
# Add import at the top
from cellvit.models.cell_segmentation.cellvit_supervit import CellViTSuperViT

# Update _get_model method
def _get_model(
    self, model_type: Literal["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI", "CellViTSuperViT"]
) -> Union[CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTSuperViT]:
    """Return the trained model for inference
    
    Args:
        model_type (str): Name of the model. Must be one of:
            CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTSuperViT
    
    Returns:
        Model instance
    """
    implemented_models = ["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI", "CellViTSuperViT"]
    if model_type not in implemented_models:
        raise NotImplementedError(
            f"Unknown model type. Please select one of {implemented_models}"
        )
    
    # ... existing code ...
    
    elif model_type == "CellViTSuperViT":
        model = CellViTSuperViT(
            model_supervit_path=None,
            num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
        )
    
    return model
```

### Step 5: Training Configuration

Create a training configuration in your dataset's config folder. Example structure:

```yaml
data:
  dataset: PanNukeDataset  # or your dataset
  dataset_path: /path/to/dataset
  num_nuclei_classes: 6
  num_tissue_classes: 19
  
model:
  backbone: SuperViT  # Identifier for your model
  embed_dim: 1536
  depth: 40
  num_heads: 24
  extract_layers: [10, 20, 30, 40]
  
training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.0001
  mixed_precision: true
  
transformations:
  normalize:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
```

---

## Model Loading and Inference

### Loading Pretrained Weights

Foundation models typically come from external sources. Here's how to load them:

```python
# Option 1: Direct state dict loading (most common)
state_dict = torch.load("path/to/checkpoint.pth", map_location="cpu")
model.encoder.load_state_dict(state_dict, strict=False)

# Option 2: Loading from HuggingFace
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    repo_id="your-org/your-model",
    filename="pytorch_model.bin"
)
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Option 3: Custom loading (e.g., for models with different key names)
state_dict = torch.load("path/to/checkpoint.pth", map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
# Rename keys if needed
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
```

### Running Inference

Once your model is integrated, use it like other CellViT models:

```bash
python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SuperViT-x40.pth \
    --outdir ./results \
    --geojson \
    process_wsi \
    --wsi_path ./data/slide.svs
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **Dimension Mismatch in Skip Connections**

**Error**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)`

**Solution**: Ensure your `extract_layers` provide features with compatible dimensions. Check that the spatial dimensions match after patch embedding and positional encoding.

```python
# Debug feature shapes
for i, layer_feat in enumerate(extracted_layers):
    print(f"Layer {i}: {layer_feat.shape}")
```

#### 2. **Patch Size Incompatibility**

**Error**: `AssertionError: Img must have a shape divisible by patch_size`

**Solution**: If your foundation model uses a different patch size (e.g., 14 instead of 16):
- Implement input rescaling (see Virchow implementation)
- Override `_forward_upsample` to handle interpolation

```python
# Example rescaling for patch_size=14
self.input_rescale_dict = {256: 252, 1024: 1022}
```

#### 3. **Missing Keys in State Dict**

**Error**: `RuntimeError: Missing key(s) in state_dict`

**Solution**: Use `strict=False` when loading and verify key names:

```python
msg = model.load_state_dict(state_dict, strict=False)
print(f"Loading result: {msg}")
# Shows missing and unexpected keys
```

#### 4. **Register Tokens (Virchow2)**

Some models use register tokens. Handle them in the forward pass:

```python
# Skip register tokens (e.g., first 4 tokens after CLS)
z4 = z4[:, 5:, :]  # Skip CLS + 4 register tokens
```

#### 5. **Custom MLP Layers**

If your foundation model uses custom activations (e.g., SwiGLU):
- Create utility classes in `utils/<your_model>_utils.py`
- Import and use in backbone definition

```python
# Example: Virchow uses SwiGLU
from cellvit.models.utils.virchow_utils import SwiGLUPacked

model = ViT(
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU,
    mlp_ratio=5.3375,
)
```

#### 6. **GPU Memory Issues**

If you encounter OOM errors:
- Reduce batch size in inference
- Update `_check_devices` in `inference_disk.py` to set appropriate limits
- Use mixed precision (`--enforce_amp`)

```python
# Update batch size limits for your model
if self.model_arch == "CellViTSuperViT":
    if gpu_memory_gb < 22:
        max_batch_size = 4
    elif gpu_memory_gb < 78:
        max_batch_size = 8
    else:
        max_batch_size = 16
```

---

## Key Architectural Differences

### Understanding Different Foundation Models

| Model | Patch Size | Embed Dim | Depth | Special Features |
|-------|-----------|-----------|-------|------------------|
| SAM-H | 16 | 1280 | 32 | Relative position encoding, global attention |
| UNI | 16 | 1024 | 24 | Standard ViT, dynamic image size |
| Virchow | 14 | 1280 | 32 | SwiGLU MLP, patch_size=14 requires rescaling |
| Virchow2 | 14 | 1280 | 32 | Register tokens (4), SwiGLU MLP |
| HIPT-256 | 16 | 384 | 12 | Smaller model for 256x256 input |

### Extract Layers Selection

Extract layers define where skip connections are taken. Guidelines:
- Need exactly **4 layers**
- Should be evenly distributed through the network depth
- Common patterns:
  - Depth 12: [3, 6, 9, 12]
  - Depth 24: [6, 12, 18, 24]
  - Depth 32: [8, 16, 24, 32]
  - Depth 40: [10, 20, 30, 40]

---

## Summary Checklist

When adding a new foundation model:

- [ ] Define model parameters (embed_dim, depth, num_heads, patch_size, extract_layers)
- [ ] Create backbone class in `backbones.py` with proper `forward()` method
- [ ] Create CellViT variant file `cellvit_<your_model>.py`
- [ ] Implement `__init__()` with proper parameter initialization
- [ ] Implement `load_pretrained_encoder()` for checkpoint loading
- [ ] Override `forward()` if custom behavior needed (e.g., rescaling, register tokens)
- [ ] Update `inference_disk.py` to include new model type
- [ ] (Optional) Update `_check_devices()` for GPU memory limits
- [ ] Test with inference pipeline
- [ ] Create training configuration
- [ ] Document model source, links, and special requirements

---

## Additional Resources

- **CellViT++ Paper**: [arXiv:2501.05269](https://arxiv.org/abs/2501.05269)
- **Original CellViT Paper**: [Medical Image Analysis 2024](https://doi.org/10.1016/j.media.2024.103143)
- **GitHub Repository**: [TIO-IKIM/CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus)
- **PyPI Package** (Inference Only): [cellvit](https://pypi.org/project/cellvit/)

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: December 2024  
**Author**: CellViT++ Development Team
