# Quick Reference: Foundation Model Integration in CellViT++

This is a quick reference guide for developers who want to integrate new foundation models into CellViT++.

## Quick Start Checklist

When adding a new foundation model, complete these steps:

### 1. Gather Model Specifications
- [ ] Embedding dimension (`embed_dim`)
- [ ] Number of transformer blocks (`depth`)
- [ ] Number of attention heads (`num_heads`)
- [ ] Patch size (`patch_size`)
- [ ] Image size (`img_size`)
- [ ] MLP ratio (if different from 4.0)
- [ ] Special features (register tokens, custom MLP, etc.)

### 2. Determine Extract Layers
- [ ] Calculate 4 evenly-spaced layers (e.g., depth=32 → [8, 16, 24, 32])
- [ ] Ensure layers are valid (1 to depth)

### 3. Create Backbone Class
File: `cellvit/models/cell_segmentation/backbones.py`

```python
class ViTCellViT<YourModel>(BaseViT):
    def __init__(self, extract_layers: List[int], ...):
        # Initialize with your model params
        
    def forward(self, x):
        # Return: (classifier_output, cls_token, extracted_layers)
```

### 4. Create CellViT Variant
File: `cellvit/models/cell_segmentation/cellvit_<yourmodel>.py`

Key methods to implement:
- `__init__()`: Set parameters, create encoder, load weights
- `load_pretrained_encoder()`: Load foundation model checkpoint
- `forward()`: Optional override if special handling needed

### 5. Update Inference Code
File: `cellvit/inference/inference_disk.py`

Add to imports and `_get_model()` method.

### 6. Test
```bash
python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-YourModel.pth \
    --outdir ./test-results \
    process_wsi \
    --wsi_path ./test_database/example.svs
```

## Code Templates

### Minimal Backbone (patch_size=16)

```python
class ViTCellViTYourModel(TimmVisionTransformer):
    def __init__(self, extract_layers: List[int], 
                 img_size: int = 224, patch_size: int = 16,
                 depth: int = 24, num_heads: int = 16,
                 embed_dim: int = 1024, num_classes: int = 0):
        super().__init__(
            img_size=img_size, patch_size=patch_size,
            depth=depth, num_heads=num_heads,
            embed_dim=embed_dim, num_classes=0
        )
        self.extract_layers = extract_layers
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor):
        extracted_layers = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)
        
        output = self.head(x[:, 0])
        return output, x[:, 0], extracted_layers
```

### Minimal CellViT Variant (Standard ViT)

```python
class CellViTYourModel(CellViT):
    def __init__(self, model_path: Union[Path, str],
                 num_nuclei_classes: int, num_tissue_classes: int,
                 drop_rate: float = 0):
        # Set parameters
        self.img_size = 224
        self.patch_size = 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.extract_layers = [6, 12, 18, 24]
        self.input_channels = 3
        self.mlp_ratio = 4.0
        
        # Call parent constructor
        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            drop_rate=drop_rate,
            regression_loss=False,
        )
        
        # Replace encoder
        self.encoder = ViTCellViTYourModel(
            extract_layers=self.extract_layers,
            num_classes=num_tissue_classes
        )
        
        # Load weights
        self.load_pretrained_encoder(model_path)
    
    def load_pretrained_encoder(self, model_path: Union[Path, str]):
        if model_path is not None:
            state_dict = torch.load(str(model_path), map_location="cpu")
            self.encoder.load_state_dict(state_dict, strict=False)
```

## Common Patterns

### Extract Layers by Depth

| Depth | Extract Layers | Pattern |
|-------|----------------|---------|
| 12 | [3, 6, 9, 12] | Every 3rd block |
| 18 | [4, 9, 13, 18] | ~Every 4-5 blocks |
| 24 | [6, 12, 18, 24] | Every 6th block |
| 32 | [8, 16, 24, 32] | Every 8th block |
| 40 | [10, 20, 30, 40] | Every 10th block |

### Skip Dimensions by Embed Dim

```python
if self.embed_dim < 512:
    self.skip_dim_11 = 256
    self.skip_dim_12 = 128
    self.bottleneck_dim = 312
else:
    self.skip_dim_11 = 512
    self.skip_dim_12 = 256
    self.bottleneck_dim = 512
```

### Loading Different Checkpoint Formats

```python
def load_pretrained_encoder(self, model_path):
    state_dict = torch.load(str(model_path), map_location="cpu")
    
    # Format 1: Direct state dict
    msg = self.encoder.load_state_dict(state_dict, strict=False)
    
    # Format 2: Nested in "model" key
    if "model" in state_dict:
        state_dict = state_dict["model"]
        msg = self.encoder.load_state_dict(state_dict, strict=False)
    
    # Format 3: Nested in "state_dict" key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        msg = self.encoder.load_state_dict(state_dict, strict=False)
    
    # Format 4: Keys need renaming
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    msg = self.encoder.load_state_dict(state_dict, strict=False)
    
    print(f"Loading checkpoint: {msg}")
```

## Special Cases

### Patch Size ≠ 16 (e.g., Virchow with patch_size=14)

Need input rescaling:

```python
# In __init__
self.input_rescale_dict = {256: 252, 1024: 1022}

# In forward()
input_shape = x.shape[2]
rescale_value = self.input_rescale_dict[input_shape]
x = F.interpolate(x, size=(rescale_value, rescale_value), mode="area")

# Later, interpolate back
b1 = F.interpolate(b1, size=(input_shape, input_shape), mode="bilinear")
```

### Register Tokens (e.g., Virchow2)

```python
# In backbone __init__
super().__init__(..., reg_tokens=4)

# In forward(), skip register tokens when reshaping
# Skip: [CLS] + [4 register tokens] = 5 tokens
z4 = z4[:, 5:, :].transpose(-1, -2).view(-1, embed_dim, *patch_dim)
```

### Custom MLP (e.g., SwiGLU)

```python
# Create utils file: cellvit/models/utils/yourmodel_utils.py
from timm.layers import SwiGLUPacked

# In backbone init
super().__init__(
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU,
    mlp_ratio=5.3375,
    ...
)
```

## Testing Your Integration

### 1. Verify Shapes
```python
# Test forward pass
model = CellViTYourModel(...)
x = torch.randn(1, 3, 1024, 1024)
out = model(x)

print("Output shapes:")
for k, v in out.items():
    print(f"{k}: {v.shape}")

# Expected:
# tissue_types: torch.Size([1, num_tissue_classes])
# nuclei_binary_map: torch.Size([1, 2, 1024, 1024])
# hv_map: torch.Size([1, 2, 1024, 1024])
# nuclei_type_map: torch.Size([1, num_nuclei_classes, 1024, 1024])
```

### 2. Check Memory
```python
# Monitor GPU memory
import torch
torch.cuda.reset_peak_memory_stats()
out = model(x)
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_mem:.2f} GB")
```

### 3. Test Inference Pipeline
```bash
# Run on test image
python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/test.pth \
    --outdir ./test-output \
    --geojson \
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
```

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Dimension mismatch | Check extract_layers match actual depth |
| Missing keys | Use `strict=False` in load_state_dict |
| OOM error | Reduce batch_size, enable --enforce_amp |
| Wrong output size | Check patch_size, add rescaling if needed |
| NaN values | Check normalization (mean, std) |

## File Locations Reference

```
cellvit/
├── models/
│   ├── cell_segmentation/
│   │   ├── backbones.py           ← Add backbone class here
│   │   ├── cellvit_<model>.py     ← Create new file
│   │   └── cellvit.py             ← Base class
│   └── utils/
│       └── <model>_utils.py       ← Optional utilities
├── inference/
│   └── inference_disk.py          ← Update _get_model()
└── training/
    └── experiments/
        └── experiment_cellvit_pannuke.py  ← Training script
```

## Additional Resources

- **Full Guide**: [docs/FOUNDATION_MODEL_GUIDE.md](FOUNDATION_MODEL_GUIDE.md)
- **CellViT++ Paper**: [arXiv:2501.05269](https://arxiv.org/abs/2501.05269)
- **Repository**: [TIO-IKIM/CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus)

---

**Quick Reference Version**: 1.0  
**Last Updated**: December 2024
