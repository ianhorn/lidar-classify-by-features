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