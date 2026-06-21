# Tutorial: Exploring a STHELAR SpatialData Object

This tutorial walks you through loading a STHELAR SpatialData `.zarr` object,
inspecting its contents, and computing a UMAP embedding from the cell-level
gene expression data.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Loading a SpatialData Object](#loading-a-spatialdata-object)
3. [Exploring the Structure](#exploring-the-structure)
4. [Accessing Cell Metadata and Gene Expression](#accessing-cell-metadata-and-gene-expression)
5. [Computing a UMAP](#computing-a-umap)
6. [Visualizing UMAP by Cell Type](#visualizing-umap-by-cell-type)
7. [Spatial Overlay of UMAP Clusters](#spatial-overlay-of-umap-clusters)

---

## Prerequisites

Install the required packages:

```bash
pip install spatialdata spatialdata-io scanpy anndata matplotlib pandas numpy
```

You will need a STHELAR SpatialData `.zarr` store. These are produced by
Stage 1 of the [MICS-Lab/STHELAR](https://github.com/MICS-Lab/STHELAR)
pipeline (`src/_1_get_sdata/process_slide.py`), or can be downloaded from
the [BioImage Archive (S-BIAD2146)](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2146).

Each `.zarr` directory contains aligned H&E imagery, Xenium spatial
transcriptomics data, CellPose segmentation masks, and aggregated transcript
counts per nucleus.

---

## Loading a SpatialData Object

```python
import spatialdata as sd

# Point to the .zarr directory for one slide
sdata_path = "/path/to/sdata_slide_id.zarr"

# Load the SpatialData object
sdata = sd.read_zarr(sdata_path)
print(sdata)
```

Expected output (structure varies by slide):

```
SpatialData object
├── Images
│   ├── 'morphology_focus': DataArray[cyx] (1, H, W)
│   └── 'HE_original': DataArray[cyx] (3, H, W)
├── Labels
│   └── 'cell_labels': DataArray[yx] (H, W)
├── Points
│   └── 'transcripts': DataFrame (N_transcripts, 4)
├── Shapes
│   └── 'cell_boundaries': GeoDataFrame (N_cells, ...)
└── Tables
    └── 'table': AnnData (N_cells, N_genes)
```

---

## Exploring the Structure

### List all elements

```python
# Images (H&E and DAPI/morphology)
print("Images:", list(sdata.images.keys()))

# Labels (segmentation masks)
print("Labels:", list(sdata.labels.keys()))

# Points (individual transcript locations)
print("Points:", list(sdata.points.keys()))

# Shapes (cell/nucleus boundaries as polygons)
print("Shapes:", list(sdata.shapes.keys()))

# Tables (cell × gene AnnData)
print("Tables:", list(sdata.tables.keys()))
```

### Inspect image dimensions

```python
# H&E image
he_image = sdata.images["HE_original"]
print(f"H&E image shape: {he_image.shape}")
# Typically (3, height, width) in CYX order

# DAPI / morphology focus image
morph = sdata.images["morphology_focus"]
print(f"Morphology image shape: {morph.shape}")
```

### View the cell table (AnnData)

```python
adata = sdata.tables["table"]
print(adata)
# AnnData object with n_obs × n_vars = N_cells × N_genes
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print(f"Observation columns: {list(adata.obs.columns)}")
```

---

## Accessing Cell Metadata and Gene Expression

### Cell annotations

```python
import pandas as pd

# Cell-level metadata lives in adata.obs
print(adata.obs.head())

# Common columns after the STHELAR annotation pipeline:
#   - 'cells_label'  → 9-class cell type (Epithelial, T_NK, Myeloid, ...)
#   - 'cells_label3' → cancer/normal binary label
#   - 'leiden'        → Leiden cluster ID
#   - 'x_centroid', 'y_centroid' → spatial coordinates

# Check available cell type annotations
if "cells_label" in adata.obs.columns:
    print("\nCell type distribution (9-class):")
    print(adata.obs["cells_label"].value_counts())
```

### Gene expression matrix

```python
import numpy as np

# Raw transcript counts per cell
X = adata.X  # sparse or dense matrix (N_cells × N_genes)
print(f"Expression matrix type: {type(X)}")
print(f"Shape: {X.shape}")

# Gene names
print(f"First 10 genes: {adata.var_names[:10].tolist()}")

# Total transcripts per cell
counts_per_cell = np.array(X.sum(axis=1)).flatten()
print(f"Median transcripts/cell: {np.median(counts_per_cell):.0f}")
```

---

## Computing a UMAP

We use [Scanpy](https://scanpy.readthedocs.io/) to compute a UMAP from the
gene expression data:

```python
import scanpy as sc

# Work on a copy to avoid modifying the original
adata_work = adata.copy()

# --- Preprocessing ---
# 1. Filter genes expressed in very few cells
sc.pp.filter_genes(adata_work, min_cells=10)

# 2. Normalize total counts per cell (library-size normalization)
sc.pp.normalize_total(adata_work, target_sum=1e4)

# 3. Log-transform
sc.pp.log1p(adata_work)

# 4. Identify highly variable genes
sc.pp.highly_variable_genes(adata_work, n_top_genes=200, flavor="seurat_v3")
print(f"Highly variable genes: {adata_work.var['highly_variable'].sum()}")

# --- Dimensionality Reduction ---
# 5. PCA on highly variable genes
sc.tl.pca(adata_work, n_comps=30, use_highly_variable=True)

# 6. Compute neighbors graph
sc.pp.neighbors(adata_work, n_neighbors=15, n_pcs=30)

# 7. Compute UMAP embedding
sc.tl.umap(adata_work)

print("UMAP coordinates stored in adata_work.obsm['X_umap']")
print(f"Shape: {adata_work.obsm['X_umap'].shape}")
```

---

## Visualizing UMAP by Cell Type

### Color by STHELAR 9-class annotation

```python
import matplotlib.pyplot as plt

# Add cell type labels to the working AnnData
if "cells_label" in adata.obs.columns:
    adata_work.obs["cell_type"] = adata.obs["cells_label"].values

    sc.pl.umap(
        adata_work,
        color="cell_type",
        title="UMAP colored by STHELAR 9-class cell type",
        frameon=False,
        save="_sthelar_celltype.png",  # saves to ./figures/umap_sthelar_celltype.png
    )
```

### Color by Leiden cluster

```python
# Run Leiden clustering on the neighbors graph
sc.tl.leiden(adata_work, resolution=0.5, key_added="leiden_new")

sc.pl.umap(
    adata_work,
    color="leiden_new",
    title="UMAP colored by Leiden clusters",
    frameon=False,
    save="_leiden.png",
)
```

### Color by total transcript count

```python
adata_work.obs["total_counts"] = counts_per_cell

sc.pl.umap(
    adata_work,
    color="total_counts",
    title="UMAP colored by total transcripts",
    frameon=False,
    cmap="viridis",
    save="_total_counts.png",
)
```

---

## Spatial Overlay of UMAP Clusters

Map the UMAP-derived clusters back onto the tissue coordinates:

```python
# Get spatial coordinates
if "x_centroid" in adata.obs.columns and "y_centroid" in adata.obs.columns:
    coords = adata.obs[["x_centroid", "y_centroid"]].values
elif "spatial" in adata.obsm:
    coords = adata.obsm["spatial"]
else:
    # Try extracting from shapes
    print("No spatial coordinates found in .obs or .obsm; check sdata.shapes")
    coords = None

if coords is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: UMAP
    umap_coords = adata_work.obsm["X_umap"]
    scatter1 = axes[0].scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=adata_work.obs["leiden_new"].astype(int),
        cmap="tab20",
        s=1,
        alpha=0.5,
    )
    axes[0].set_title("UMAP (Leiden clusters)")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")

    # Right: Spatial coordinates colored by cluster
    scatter2 = axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=adata_work.obs["leiden_new"].astype(int),
        cmap="tab20",
        s=0.5,
        alpha=0.3,
    )
    axes[1].set_title("Spatial map (Leiden clusters)")
    axes[1].set_xlabel("X coordinate (pixels)")
    axes[1].set_ylabel("Y coordinate (pixels)")
    axes[1].invert_yaxis()  # Match histology orientation

    plt.tight_layout()
    plt.savefig("spatial_umap_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()
```

---

## Summary

| Step | What you learned |
|------|-----------------|
| Load | `sd.read_zarr()` opens a STHELAR `.zarr` store as a `SpatialData` object |
| Explore | Images, labels, points, shapes, and tables are all accessible as named elements |
| AnnData | The cell × gene table is a standard `AnnData` object in `sdata.tables["table"]` |
| UMAP | Standard Scanpy workflow: normalize → HVG → PCA → neighbors → UMAP |
| Visualize | Color UMAP by cell type, cluster, or any continuous variable |
| Spatial | Map clusters back to tissue coordinates for spatial context |

## Next Steps

- **Marker genes**: Use `sc.tl.rank_genes_groups()` to find genes that
  distinguish each cluster or cell type.
- **Spatial statistics**: Use `squidpy` for spatial neighborhood analysis
  (`sq.gr.spatial_neighbors`, `sq.gr.nhood_enrichment`).
- **Multi-slide analysis**: Load multiple `.zarr` files, concatenate their
  AnnData tables, and compute a joint UMAP to compare tissue types.
- **Integration with CellViT++**: After running inference with `detect_cells.py`,
  load the resulting cell embeddings (`.pt` files) and project them alongside
  the transcriptomic UMAP to compare morphological vs. molecular similarity.
