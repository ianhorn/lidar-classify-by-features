# Classify Lidar by Feature Extraction / Machine Learning

Classifies building points in Kentucky's [KyFromAbove](https://kyfromabove.ky.gov/) Phase 2 lidar. Each point gets per-point geometric and contextual features, an XGBoost model is trained against Overture Maps building footprints, and the model's predictions are written back per tile.

>Primary objective - classify building points  
>Secondary objective - a feature extraction product that can classify other entities

## Pipeline

Everything lives in the `lidar-classification` S3 bucket under `phase2/`.

**Phase 1 - feature extraction** (`src/`), one tile at a time:

1. `create_buffered_tile.py` - buffer the tile's bbox, search STAC for neighboring COPC tiles, and crop their points into a temporary buffered tile (so edge points get full neighborhoods).
2. Join Overture building footprints (buffered 30 m) to get the `WithinFootprint` label.
3. `calculate_point_features.py` - PDAL `filters.hag_delaunay` plus PCA/KDTree neighborhood features.
4. Upload `laz/`, `buildings/` and `features/` outputs to S3.

Drivers: `main.py` (Dask on Coiled), `run_phase1_batch.py` (AWS Batch array job, used after Coiled credits ran out), and `run_sync_batch.py` (re-process tiles that changed upstream).

**Phase 2 - classification** (`scripts/`):

- `train_building_classifier.py` - train on a stratified sample of feature parquets (`WithinFootprint` is the label). Models are saved as `building_classifier*.json`; v4 is in production.
- `classify_tiles.py` - apply the model to every feature tile, writing `IsBuildingML` to `phase2/classified/`. Every point is classified, including those the source tagged noise (class 7/18). `IsBuildingML` also requires `HAG > --hag-floor`.
- `analyze_false_positives.py` - read-only diagnostics on where the model disagrees with the footprint label.

## Features

Defined in `FEATURE_COLUMNS` in `scripts/common.py`. X/Y/Z, the source `Classification`, and the label are deliberately excluded.

| Feature | Source |
|---|---|
| `HAG` | PDAL `filters.hag_delaunay` |
| `Planarity`, `Linearity`, `Sphericity`, `SurfaceVariation`, `Roughness`, `Verticality`, `Normal{X,Y,Z}`, `NeighborCount`, `Density` | Neighborhood geometry |
| `Intensity` | Source LAS |
| `ReturnNumber`, `NumberOfReturns` | Source LAS; helps separate canopy (multi-return) from roofs |
| `DistToBridge`, `InBridgeBuffer` | KYTC bridge inventory (`bridge_features.py`); filters bridge-deck false positives |
| `CanopyHeight` | Meta/WRI canopy height raster (`canopy_features.py`) |

The last three groups were added after phase 1, so `backfill_return_number.py`, `backfill_bridge_features.py` and `backfill_canopy_features.py` rewrite existing feature parquets on S3 instead of recomputing everything.

## Keeping outputs in sync with upstream

The bucket listing, not the STAC catalog, is the source of truth for which tiles exist (the STAC lags and has stale `.copz` ids).

1. `diff_stac_changes.py` - compare the bucket to our outputs; writes a manifest of `new` / `replaced` / `retry` tiles and an orphan list to `stac_stage/`.
2. `apply_stac_changes.py` - `--stage` uploads the manifest, `--delete-orphans` and `--refresh-full-list` clean up. Destructive steps are dry runs unless `--execute`.
3. `run_sync_batch.py` - process the manifest on AWS Batch.

Housekeeping: `refresh_todo_list.py` rebuilds the todo list (`stac_item_list.parquet`) from `stac_item_list_full.parquet`, `prune_stale_problematic.py` clears `laz-problematic/` flags for tiles that finished anyway, and `tile_issues/` lists tiles with bad headers or that are unprocessable (e.g. water bodies).

## Running at scale

- **Coiled**: `src/main.py`, looped by `scripts/continuous_run.sh`. Environment built by `create_coiled_environment.py`.
- **AWS Batch**: `docker/Dockerfile` (CUDA runtime base, one image for CPU and GPU job definitions). Phase 1 and backfills use CPU jobs; `classify_tiles.py` uses a GPU job. About 112 very dense tiles have feature parquets over 4 GB and OOM at 30 GB / 4 vCPU, so use `--workers 1` or the high-memory job definition for full-tile-set runs.

## Environment

Linux/Mac/Windows
```bash
git clone https://github.com/ianhorn/lidar-classify-by-features.git
cd lidar-classify-by-features

mamba env create -f environment.yml
mamba activate lidar-classification
```

AWS credentials with access to the `lidar-classification` bucket are needed for anything that touches S3.
