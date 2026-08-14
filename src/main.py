#!/usr/bin/env python3

############################################################
#               RUN DATAFRAME THROUGH COILED              #
############################################################

from coiled import Cluster
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from pathlib import Path
import create_buffered_tile as cbt
import calculate_point_features as cpf
from concurrent.futures import ThreadPoolExecutor


import tempfile
import boto3
import time

S3_BUCKET = "lidar-classification"


def upload_to_s3(local_path, bucket, key):
    """Upload a local file to S3."""

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded {local_path} to s3://{bucket}/{key}")


start_time = time.time()
df = pd.read_parquet('s3://lidar-classification/phase2/stac_item_list.parquet')
df.shape


############################################################
#                     CREATE VARIABLES                    #
############################################################

# add buildings folder that will store building parquet files
buildings = Path('buildings')
buildings.mkdir(exist_ok=True)

# add a laz folder that will temporarily hold laz files
laz_path = Path('laz')
laz_path.mkdir(exist_ok=True)

# add a lidar features folders that will hold feature parquet files
lidar_features = Path('lidar_features')
lidar_features.mkdir(exist_ok=True)


############################################################
#                    CREATE A CLUSTER                      #
############################################################

cluster = Cluster(
    software='lidar-classification',
    n_workers=150,
)

client = cluster.get_client()
client.upload_file("src/create_buffered_tile.py")
client.upload_file("src/calculate_point_features.py")


############################################################
#                   PROCESS ITEM FUNCTION                  #
############################################################

def process(stac_item):

    # stac item
    item_id = stac_item

    try:
        lasfile = Path(f'lidar-file/{item_id}.laz')
        collection = 'laz-phase2'
        stac = 'https://spved5ihrl.execute-api.us-west-2.amazonaws.com'

        # stac call
        item = cbt.get_stac_item(item_id, collection, stac)
        bbox = item.bbox
        print(f'bbox {bbox}')

        # increase bbox by 30 meter buffer
        buffer = 30
        distance = cbt.get_distance_degrees(buffer)
        bbox_buffer = cbt.get_buffered_bbox
        print(f'Degrees: {distance}')
        hrefs = cbt.search_stac(stac, collection, bbox_buffer)

        # repoject buffered bbox
        bbox_3089 = cbt.reproject_bbox(bbox_buffer)
        print(f'3089 bbox: {bbox_3089}')

        # calculate bounds used for cropping later
        bounds = cbt.pdal_bounds(bbox_3089)

        # state the OvertureMaps release
        overture_release = '2026-07-22.0'
        overture_url = f's3://overturemaps-us-west-2/release/{overture_release}/theme=buildings/type=building/*'

        # crop_copc and get_buffered_tile_footprints don't depend on each other's
        # output (both only need bbox_buffer/bounds), and both are dominated by
        # network wait (S3), so run them on separate threads instead of back to back.
        with ThreadPoolExecutor(max_workers=2) as executor:
            crop_future = executor.submit(cbt.crop_copc, hrefs, bounds, lasfile)
            footprints_future = executor.submit(cbt.get_buffered_tile_footprints, item, overture_url, bbox_buffer)

            crop_future.result()
            tile_buildings = footprints_future.result()

        print(f'{tile_buildings} created')

        # create building footprints file
        footprints_file = f'building-files/{item.parquet}.parquet'
        features_file = f'lidar-features/{item.id}.parquet'
        radius = 7.0
        chunk_size = 90_000

        # load points
        points = cpf.load_points(lasfile)
        hag = points['HeightAboveGround']

        # create a numpy array
        xyz = np.column_stack([points["X"], points["Y"], points["Z"]])
        tree = cKDTree(xyz)

        # compute geometric features
        start = time.perf_counter()
        geom_vec = cpf.compute_geometric_features_vectorized(tree, xyz, radius=radius, chunk_size=chunk_size)
        print(f'geometric features: {time.perf_counter() - start:.1f}s')

        # Determine if points are within footrpint
        within_footprint, footprint_height = cpf.compute_within_footprint(xyz, footprints_file)
        geom_vec["WithinFootprint"] = within_footprint
        geom_vec["FootprintHeight"] = footprint_height

        # export features to parquet
        cpf.export_features(xyz, points, hag, geom_vec, features_file)
        print(f'Delete features file "{features_file}"')
        Path(features_file).unlink(missing_ok=True)
        print(f'"{features_file}" deleted.')

        return item_id, "ok", ""
    except Exception as e:
        print(f'ERROR processing {item_id}: {e}')
        return item_id, "error", str(e)


############################################################
#                         RUN LOOP                        #
############################################################

futures = []
for _, row in df.iterrows():
    future = client.submit(process, row[0])
    futures.append(future)

results = client.gather(futures)

results_df = pd.DataFrame(results, columns=["item_id", "status", "error"])
local_path = Path(tempfile.gettempdir()) / "future_results.parquet"
results_df.to_parquet(local_path)
upload_to_s3(local_path, S3_BUCKET, "phase2/future_results.parquet")
local_path.unlink()
