#!/usr/bin/env python3

from coiled import Cluster
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from pathlib import Path
import create_buffered_tile as cbt
import calculate_point_features as cpf

import time

start_time = time.time()
df = pd.read_parquet('s3://lidar-classification/phase2/stac_item_list.parquet')
print(df)

# add buildings folder that will store building parquet files
buildings = Path('buildings')
buildings.mkdir(exist_ok=True)

# add a laz folder that will temporarily hold laz files
laz_path = Path('laz')
laz_path.mkdir(exist_ok=True)

# add a lidar features folders that will hold feature parquet files
lidar_features = Path('lidar_features')
lidar_features.mkdir(exist_ok=True)


# Create a cluster 
cluster = Cluster(
    software='lidar-classification',
    n_workers=1000,
)

def process(item):
    lasfile = item
    footprints_file = f'building-files/{item.parquet}.parquet'
    features_file = f'lidar-features/{item.id}.parquet'
    radius = 7.0
    chunk_size = 90_000

    points = cpf.load_points(lasfile)
    hag = points['HeightAboveGround']

    xyz = np.column_stack([points["X"], points["Y"], points["Z"]])
    tree = cKDTree(xyz)

    start = time.perf_counter()
    geom_vec = cpf.compute_geometric_features_vectorized(tree, xyz, radius=radius, chunk_size=chunk_size)
    print(f'geometric features: {time.perf_counter() - start:.1f}s')

    within_footprint, footprint_height = cpf.compute_within_footprint(xyz, footprints_file)
    geom_vec["WithinFootprint"] = within_footprint
    geom_vec["FootprintHeight"] = footprint_height

    cpf.export_features(xyz, points, hag, geom_vec, features_file)

    print(f'Delete features file "{features_file}"')
    features_file.unlink(existing_ok=True)
    print(f'"{features_file}" deleted.')