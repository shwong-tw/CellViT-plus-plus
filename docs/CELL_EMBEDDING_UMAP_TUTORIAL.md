# Tutorial: Extract Cell Embeddings and Visualize in UMAP Space

This tutorial shows how to:

1. Run CellViT++ inference and export per-cell embeddings
2. Load saved embeddings from `*_cells.pt`
3. Visualize embeddings in a 2D UMAP projection

## 1) Run inference and export embeddings

Use the `--graph` flag to export a PyTorch graph file that includes cell embeddings:

```bash
python3 ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --outdir ./results/embedding_tutorial \
  --graph \
  process_wsi \
  --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
```

After inference, you should find:

- `./results/embedding_tutorial/JP2K-33003-2_cells.pt` (embeddings + positions + metadata)
- `./results/embedding_tutorial/JP2K-33003-2_cells.json` (cell class predictions)

## 2) Install UMAP dependency

`umap-learn` is not pinned in the default requirements file, so install it once in your environment:

```bash
pip install umap-learn
```

## 3) Load embeddings and run UMAP

Create a script (for example `/tmp/umap_cell_embeddings.py`) with:

```python
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


outdir = Path("./results/embedding_tutorial")
stem = "JP2K-33003-2"

graph = torch.load(outdir / f"{stem}_cells.pt", map_location="cpu")
embeddings = graph.x.cpu().numpy()  # shape: [num_cells, embedding_dim]
positions = graph.positions.cpu().numpy()  # shape: [num_cells, 2]

with open(outdir / f"{stem}_cells.json", "r") as f:
    cell_json = json.load(f)

label_map = {int(k): v for k, v in cell_json["type_map"].items()}
cell_types = np.array([c["type"] for c in cell_json["cells"]], dtype=int)

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
embedding_2d = reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
for type_id in np.unique(cell_types):
    mask = cell_types == type_id
    plt.scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        s=3,
        alpha=0.6,
        label=label_map.get(int(type_id), f"type_{type_id}"),
    )

plt.title("CellViT++ cell embeddings in UMAP space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(markerscale=4, fontsize=8)
plt.tight_layout()
plt.savefig(outdir / f"{stem}_umap.png", dpi=300)
plt.show()
```

Run it:

```bash
python3 /tmp/umap_cell_embeddings.py
```

## 4) Multi-slide embedding visualization

You can combine multiple `*_cells.pt` files before UMAP to compare slides in one shared space.

Recommended workflow:

1. Concatenate embeddings from all slides into one matrix
2. Keep one metadata column (`slide_id`) to color points by slide
3. Optionally subsample cells per slide (for example 20k–100k cells per slide) for faster plotting

## 5) Notes and interpretation tips

- `graph.x` stores the cell embedding vectors.
- `graph.positions` stores cell centroids in slide coordinates.
- If classes overlap strongly in UMAP, try different `n_neighbors`, `min_dist`, or `metric`.
- UMAP is stochastic; keep `random_state` fixed for reproducible plots.
