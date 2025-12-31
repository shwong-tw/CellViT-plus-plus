# Foundation Model Training Configurations

This directory contains example configuration files for training CellViT++ with different foundation models.

## Available Configurations

| File | Foundation Model | Memory (VRAM/image) | Batch Size (24GB GPU) | Description |
|------|-----------------|---------------------|----------------------|-------------|
| [`train_sam_h_pannuke.yaml`](train_sam_h_pannuke.yaml) | SAM-H | ~6GB | 4 | Largest SAM variant (baseline) |
| [`train_uni_pannuke.yaml`](train_uni_pannuke.yaml) | UNI | ~4GB | 8 | Pathology-specific foundation model |
| [`train_virchow_pannuke.yaml`](train_virchow_pannuke.yaml) | Virchow | ~5GB | 6 | Paige.AI foundation model |

## Quick Start

### 1. Choose Your Foundation Model

Download the checkpoint for your chosen foundation model:

- **SAM-H**: [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- **UNI**: [HuggingFace](https://huggingface.co/mahmoodlab/UNI)
- **Virchow**: [HuggingFace](https://huggingface.co/paige-ai/Virchow)

Place checkpoints in `./checkpoints/` directory.

### 2. Update Configuration

Edit the chosen config file and update these paths:

```yaml
data:
  dataset_path: /path/to/your/dataset  # Update this

model:
  pretrained_encoder: ./checkpoints/your_model.pth  # Update this
```

### 3. Run Training

```bash
# Example with UNI
python3 ./cellvit/train_cellvit.py \
  --config ./configs/foundation_models/train_uni_pannuke.yaml
```

## Comparing Multiple Models

To compare different foundation models on the same dataset:

### Option 1: Sequential Training

```bash
#!/bin/bash
# Train with each foundation model

for config in train_sam_h_pannuke.yaml train_uni_pannuke.yaml train_virchow_pannuke.yaml; do
  echo "Training with $config"
  python3 ./cellvit/train_cellvit.py \
    --config "./configs/foundation_models/$config"
done
```

### Option 2: Parallel Training (if you have multiple GPUs)

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 ./cellvit/train_cellvit.py \
  --config ./configs/foundation_models/train_sam_h_pannuke.yaml

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python3 ./cellvit/train_cellvit.py \
  --config ./configs/foundation_models/train_uni_pannuke.yaml

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python3 ./cellvit/train_cellvit.py \
  --config ./configs/foundation_models/train_virchow_pannuke.yaml
```

## Configuration Parameters Explained

### Key Parameters to Modify

1. **Foundation Model Selection**:
   ```yaml
   model:
     backbone: UNI  # Options: sam-h, sam-l, sam-b, uni, virchow, virchow2
     pretrained_encoder: /path/to/checkpoint
   ```

2. **Batch Size** (adjust based on GPU memory):
   ```yaml
   training:
     batch_size: 8  # Decrease if OOM, increase if memory available
   ```

3. **Dataset Paths**:
   ```yaml
   data:
     dataset_path: /path/to/dataset
     train_folds: [0]  # Which fold to use for training
     val_folds: [1]    # Which fold to use for validation
   ```

4. **Logging**:
   ```yaml
   logging:
     mode: online     # online, offline, or disabled
     project: My-Project-Name
     log_comment: My-Experiment-Name
   ```

### Memory Optimization

If you encounter out-of-memory errors:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 4  # or even 2
   ```

2. **Enable mixed precision** (already enabled in configs):
   ```yaml
   training:
     mixed_precision: true
   ```

3. **Use gradient accumulation** (add to config):
   ```yaml
   training:
     gradient_accumulation_steps: 2  # Effective batch size = batch_size * accumulation_steps
   ```

## Expected Training Time

On a single A100 80GB GPU with PanNuke dataset:

| Model | Batch Size | Time per Epoch | Total Time (130 epochs) |
|-------|-----------|----------------|------------------------|
| SAM-H | 8 | ~15 min | ~32 hours |
| UNI | 16 | ~15 min | ~32 hours |
| Virchow | 12 | ~18 min | ~39 hours |

*Note: Times may vary based on hardware and dataset size*

## Evaluation and Comparison

After training, compare models using:

1. **WandB Dashboard**: All models are logged to the same project for easy comparison
2. **Validation Metrics**: Check `Dice Score`, `F1 Score`, `AJI` in logs
3. **Inference Speed**: Test with `detect_cells.py` on the same WSI

Example inference comparison:

```bash
# SAM-H
python3 ./cellvit/detect_cells.py \
  --model ./logs/CellViT-SAM-H-Fold-1/checkpoints/model_best.pth \
  --outdir ./results/sam_h \
  process_wsi --wsi_path ./test.svs

# UNI
python3 ./cellvit/detect_cells.py \
  --model ./logs/CellViT-UNI-Fold-1/checkpoints/model_best.pth \
  --outdir ./results/uni \
  process_wsi --wsi_path ./test.svs

# Compare outputs in ./results/
```

## Customization

To create your own configuration:

1. Copy one of the example files
2. Modify parameters as needed
3. Key sections to customize:
   - `data`: Dataset paths and splits
   - `model`: Foundation model and architecture
   - `training`: Hyperparameters and batch size
   - `loss`: Loss function weights (if needed)

## Additional Foundation Models

To use other foundation models (SAM-B, SAM-L, Virchow2):

1. Create a new config file based on an existing one
2. Change the `backbone` parameter:
   ```yaml
   model:
     backbone: sam-b  # or sam-l, virchow2
   ```
3. Update the checkpoint path
4. Adjust batch size based on model size

## Troubleshooting

### Issue: Training starts but crashes after a few batches

**Solution**: Reduce batch size
```yaml
training:
  batch_size: 2  # Start small and increase
```

### Issue: Checkpoint not found

**Solution**: Verify paths
```bash
ls -lh ./checkpoints/  # Check if checkpoint exists
```

### Issue: Different performance than paper

**Possible reasons**:
- Different random seed
- Different dataset preprocessing
- Different loss weights
- Different number of training epochs

Make sure all hyperparameters match the paper for reproducibility.

## Further Reading

- **[Foundation Model Guide](../../docs/FOUNDATION_MODEL_GUIDE.md)**: Comprehensive documentation
- **[Quick Start](../../docs/QUICK_START_FOUNDATION_MODELS.md)**: Fast track guide
- **[Main README](../../README.md)**: Project overview and setup

## Support

For issues or questions:
1. Check the full documentation in `./docs/`
2. Review existing training logs in `./logs/`
3. Open an issue on GitHub with your configuration and error logs
