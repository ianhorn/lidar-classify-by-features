# Classify Lidar by Feature Extraction / Machine Learning

This creates a workflow that will take a single AOI, for the purposes here, a Lidar point cloud tile bbox.  Runs a buffer, uses stac to search the bbox.  Creates a temporary point cloud file of the buffered bbox area.  

Refer to [pseudocode](PSEUDOCODE.md) for flowchart

>Primary objective - classify building points  
>secondary objective a feature extraction product that can classify other entities

## Features to extract

 - HAG - height above ground, likely using PDAL with `"filters": "hag_delaunay"`
 - planarity
 - verticality
 - within a building polygon footprint (OvertureMaps)
 - density 
 - roughness
 - more to be decided

## Environment

Linux/Mac/Windows
```bash
git clone https://github.com/ianhorn/lidar-classify-by-features.git
cd lidar-classify-by-features

mamba create -f environment.yml
```

## Feature extraction

The merged LAS/LAZ file created by the buffered-tile workflow can be passed directly into the feature extractor.

The buffered-tile workflow also writes a GeoParquet file of Overture building footprints to a `vector_geometries/` folder next to the merged LAS/LAZ output. That file is exported using the same buffered bounding box used for the STAC search.

```bash
python src/calculate_vars.py /path/to/merged_tile.laz /path/to/output.csv \
  --footprint-file /path/to/buildings.geojson \
  --neighbor-radius 0.5
```

The script uses PDAL for HAG and normal estimation, then adds point-level features:
- `hag`
- `planarity`
- `roughness`
- `density`
- `verticality`
- `inside_footprint`

If you do not have building footprints available yet, omit `--footprint-file` and the script will still emit the other features.
