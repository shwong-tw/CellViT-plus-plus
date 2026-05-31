# Tutorial: Public Datasets and Classifier Training

This guide summarizes the public dataset loaders in:

- `./cellvit/training/datasets/`

and explains how to train a CellViT classifier head with each dataset.

---

## 1) Training entrypoints

Main classifier training scripts:

- Standard classifier head:
  - `python3 ./cellvit/train_cell_classifier_head.py --config /path/to/config.yaml`
- Lizard + histomics features:
  - `python3 ./cellvit/train_cell_classifier_head_histomics.py --config /path/to/config.yaml`

The config parser also supports:

- `--sweep` for WandB sweep
- `--agent <entity/project/sweep_id>` to attach an agent
- `--checkpoint <path>` to resume

---

## 2) Dataset support overview

### Directly supported by `train_cell_classifier_head.py`

- **Ocelot** (`OcelotDataset`)
- **CoNSeP** (`CoNSePDataset`)
- **Lizard (pre-extracted graphs)** (`LizardGraphDataset`, dataset key: `lizard_preextracted`)
- **MIDOG** (`MIDOGDataset`)
- **NuCLS** (`NuCLSDataset`)
- **PanopTILs** (`PanoptilsDataset`)
- **SegPath** (`SegPathDataset`)

### Supported with dedicated entrypoint

- **Lizard histomics** (`LizardHistomicsDataset`)
  - Use `train_cell_classifier_head_histomics.py`
  - Dataset key: `lizard_histomics`

### Public datasets in `training/datasets` that are **not directly wired** into classifier-head experiment

- **PanNuke** (`PanNukeDataset`) – segmentation training dataset
- **MoNuSeg** (`MoNuSegDataset`) – segmentation training dataset

For classifier-head training on these, use the generic classifier loaders:

- `DetectionDataset` (point annotations as CSV)
- `SegmentationDataset` (HoVer-Net style `.npy` with `inst_map` + `type_map`)

---

## 3) Common config skeleton (classifier head)

```yaml
logging:
  mode: offline
  project: cellvit++
  notes: classifier-training
  log_comment: my-run
  wandb_dir: ./logs_local/wandb
  log_dir: ./logs_local
  level: Info

random_seed: 19
gpu: 0

data:
  dataset: Ocelot
  dataset_path: /path/to/dataset
  train_filelist: /path/to/train.csv        # dataset-dependent
  val_filelist: /path/to/val.csv            # dataset-dependent
  num_classes: 2

cellvit_path: ./checkpoints/CellViT-SAM-H-x40-AMP.pth

model:
  hidden_dim: 128

training:
  cache_cell_dataset: true
  batch_size: 256
  epochs: 50
  drop_rate: 0.1
  optimizer: AdamW
  optimizer_hyperparameter:
    lr: 0.001
    weight_decay: 0.0001
    betas: [0.85, 0.9]
  early_stopping_patience: 20
  mixed_precision: true
  eval_every: 1
  scheduler:
    scheduler_type: exponential
```

Then run:

```bash
python3 ./cellvit/train_cell_classifier_head.py --config /path/to/config.yaml
```

---

## 4) Per-dataset tutorial

## Ocelot

Loader: `cellvit/training/datasets/ocelot.py`

Expected structure:

```text
<dataset_path>/
  images/
    train/cell/*.jpg
    val/cell/*.jpg
    test/cell/*.jpg
  annotations/
    train/cell/*.csv
    val/cell/*.csv
    test/cell/*.csv
```

CSV format: `x,y,type` per row (`type` is read and converted to zero-based classes in loader).

Config keys:

- `data.dataset: Ocelot`
- `data.dataset_path`
- optional `data.train_filelist` (subset selection)

Reference assets:

- `./logs/Datasets/Ocelot/`
- `./logs/Classifiers/Ocelot/`

---

## CoNSeP

Loader: `cellvit/training/datasets/consep.py`

Expected structure:

```text
<dataset_path>/
  Train/
    images/*.png
    detections/*.json
  Test/
    images/*.png
    detections/*.json
  splits/
    fold_*/train.csv
    fold_*/val.csv
```

Notes:

- Loader expects split-level directories (`Train`, `Test`).
- For train/val inside Train, pass `train_filelist` and `val_filelist`.

Config keys:

- `data.dataset: CoNSeP`
- `data.dataset_path`
- `data.train_filelist` and `data.val_filelist` (required for split training)

Reference assets:

- `./logs/Datasets/CoNSeP/`
- `./logs/Classifiers/CoNSeP/`

---

## Lizard (pre-extracted CellViT graphs)

Loader: `cellvit/training/datasets/lizard.py` (`LizardGraphDataset`)

Expected structure:

```text
<dataset_path>/
  fold_1/
    labels/*.mat
    predictions-cellvit/<network_name>/
      *_cells.pt
      *_cells.json
  fold_2/
  fold_3/
```

`<network_name>` is one of `SAM-H`, `UNI`, `ViT256`.

Config keys:

- `data.dataset: lizard_preextracted`
- `data.dataset_path`
- `data.train_fold` (e.g. `fold_1`)
- `data.val_fold` (e.g. `fold_2`)
- `data.network_name`

Reference assets:

- `./logs/Datasets/Lizard/`
- `./logs/Classifiers/Lizard/`

---

## Lizard (histomics features)

Loader: `LizardHistomicsDataset` in `cellvit/training/datasets/lizard.py`  
Entrypoint: `train_cell_classifier_head_histomics.py`

Extra requirement:

```text
<dataset_path>/<train_fold>/norm-vectors/<network_name>/mean.npy
<dataset_path>/<train_fold>/norm-vectors/<network_name>/std.npy
```

Config keys:

- `data.dataset: lizard_histomics`
- `data.dataset_path`
- `data.train_fold`, `data.val_fold`
- `data.network_name`

Command:

```bash
python3 ./cellvit/train_cell_classifier_head_histomics.py --config /path/to/config.yaml
```

---

## MIDOG

Loader: `cellvit/training/datasets/midog.py`

Expected structure:

```text
<dataset_path>/
  images/*.tiff
  midog.json
```

Filelists contain image filenames used for train/val subsets.

Config keys:

- `data.dataset: MIDOG`
- `data.dataset_path`
- `data.train_filelist` and `data.val_filelist`
- commonly used trainer extras:
  - `data.gt_json_path`
  - `data.cell_graph_path`
  - `data.x_valid_path`

Reference assets:

- `./logs/Datasets/MIDOG++/`
- `./logs/Classifiers/MIDOG/`

---

## NuCLS

Loader: `cellvit/training/datasets/nucls.py`

Expected structure:

```text
<dataset_path>/
  train/
    images/*.png
    labels/*.csv
  val/
    images/*.png
    labels/*.csv
  test/
    images/*.png
    labels/*.csv
  splits/
    fold_*/train.csv
    fold_*/val.csv
```

Important label setting:

- `data.classification_level` must match one of:
  - `raw_classification`
  - `main_classification`
  - `super_classification`

Config keys:

- `data.dataset: NuCLS` (or `nucls_label` for label-aware trainer variant)
- `data.dataset_path`
- `data.train_filelist`, `data.val_filelist`
- `data.classification_level`

Reference assets:

- `./logs/Datasets/NuCLS/`
- `./logs/Classifiers/NuCLS/`

---

## PanopTILs

Loader: `cellvit/training/datasets/panoptils.py`

Expected structure:

```text
<dataset_path>/
  train/
    images/*.png
    annotations/*.csv
    splits/fold_*/train.csv
    splits/fold_*/val.csv
  test/
    images/*.png
    annotations/*.csv
```

CSV format: `x,y,type` rows.

Config keys:

- `data.dataset: PanopTILs`
- `data.dataset_path`
- `data.train_filelist`, `data.val_filelist`
- optional class balancing:
  - `training.weighted_sampling: true`
  - `training.weight_list: [...]`

Reference assets:

- `./logs/Datasets/Panoptils/`
- `./logs/Classifiers/PanopTILs/`

---

## SegPath

Loader: `cellvit/training/datasets/segpath.py`

Expected structure:

```text
<dataset_path>/
  <sample>_HE.png
  <sample>_mask.png
```

Filelists should contain `<sample>` stem names (without `_HE`).

Config keys:

- `data.dataset: SegPath`
- `data.dataset_path`
- `data.train_filelist`, `data.val_filelist` (required)
- `data.ihc_threshold`
- `data.input_shape` (typically 960)

Reference assets:

- `./logs/Datasets/SegPath/`
- `./logs/Classifiers/SegPath/`

---

## PanNuke

Loader: `cellvit/training/datasets/pannuke.py`  
Primary usage: segmentation model training/evaluation (not directly instantiated by `train_cell_classifier_head.py`).

If you want classifier-head training from PanNuke-style labels:

1. Convert to generic classifier format:
   - detection points (`DetectionDataset`) or
   - HoVer-Net `.npy` dict labels (`SegmentationDataset`)
2. Train with:
   - `data.dataset: DetectionDataset` or `data.dataset: SegmentationDataset`

---

## MoNuSeg

Loader: `cellvit/training/datasets/monuseg.py`  
Primary usage: segmentation workflow.

For classifier-head training:

1. Export nucleus centroids and classes into `DetectionDataset` format, or
2. Convert labels into `SegmentationDataset` format (`inst_map` + `type_map`)
3. Train with `train_cell_classifier_head.py`

---

## 5) Practical training workflow (recommended)

1. Prepare one dataset exactly matching its loader format.
2. Start from a proven config in:
   - `./logs/Classifiers/<Dataset>/.../config.yaml`
3. Update:
   - `data.dataset_path`
   - split filelists/folds
   - `cellvit_path`
   - `num_classes` and `label_map`
4. Run one non-sweep training:
   - `python3 ./cellvit/train_cell_classifier_head.py --config /path/to/config.yaml`
5. (Optional) run sweep:
   - `python3 ./cellvit/train_cell_classifier_head.py --config /path/to/sweep.yaml --sweep`
