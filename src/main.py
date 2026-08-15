#!/usr/bin/env python3

############################################################
#               RUN DATAFRAME LOCALLY, SEQUENTIALLY       #
############################################################

# No Coiled/Dask here -- this branch runs on a single local machine instead
# of a cloud cluster, since Coiled credits run out after ~20 hours/month.
# process() is unchanged from the Coiled version: it was already
# environment-agnostic (plain boto3 S3 calls, relative local paths, no
# Coiled-specific code inside it), so nothing about *how a tile gets
# processed* needed to change -- only how tiles get scheduled.
#
# Sequential on purpose, not a local process pool: a single tile's feature
# computation was measured peaking at ~23GiB RAM on Coiled (one of the
# denser buffered tiles, ~33M points). Running more than one of those
# concurrently on a 32GB machine risks OOM with no equivalent to Dask's
# pause/resume memory manager to catch it locally. DuckDB and NumPy/OpenBLAS
# already parallelize across all local cores on their own for a single tile
# (confirmed: DuckDB's default thread count and OpenBLAS both already use
# all 24 cores here without any config from us), so going sequential doesn't
# leave meaningful speed on the table -- it's a memory-safety choice, not a
# parallelism tradeoff.

# Cap at 20 of this machine's 24 cores, reserving 4 so the computer stays
# usable for everything else while this runs for hours. Must be set before
# numpy/create_buffered_tile/calculate_point_features are imported --
# OpenBLAS reads OPENBLAS_NUM_THREADS at first use, and
# calculate_point_features reads LIDAR_LOCAL_MAX_WORKERS at import time.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["LIDAR_LOCAL_MAX_WORKERS"] = "20"

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
from botocore.exceptions import ClientError

S3_BUCKET = "lidar-classification"


def upload_to_s3(local_path, bucket, key):
    """Upload a local file to S3."""

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded {local_path} to s3://{bucket}/{key}")


def s3_object_exists(bucket, key):
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


start_time = time.time()
# Local copy, not S3 -- avoids re-downloading the same 1.9MB file every run,
# and lets this run fully offline apart from the actual STAC/S3/Overture
# calls inside process(). Same file, just already sitting at the repo root.
df = pd.read_parquet('stac_item_list.parquet')

# The Coiled run (still active separately) works through df top-down.
# Reversing here means this local run starts from the opposite end, so the
# two converge toward the middle instead of one redoing the other's work --
# though the S3 skip-check in process() would catch that anyway.
df = df.iloc[::-1]
df.shape


############################################################
#                     CREATE VARIABLES                    #
############################################################

# These names must match the hardcoded path prefixes used inside process()
# (lasfile, footprints_file, features_file) -- a mismatch here means the
# directory a worker tries to write into doesn't exist, which crashes the
# whole worker process (PDAL's write failure throws an uncaught C++
# exception), not just the one task.

# add buildings folder that will store building parquet files
buildings = Path('building-files')
buildings.mkdir(exist_ok=True)

# add a laz folder that will temporarily hold laz files
laz_path = Path('laz-files')
laz_path.mkdir(exist_ok=True)

# add a lidar features folders that will hold feature parquet files
lidar_features = Path('lidar-features')
lidar_features.mkdir(exist_ok=True)


############################################################
#                   PROCESS ITEM FUNCTION                  #
############################################################

def process(stac_item):

    # stac item
    item_id = stac_item
    if 'copz' in item_id:
        item_id = item_id.replace('copz', 'copc')

    try:
        # .laz (LASzip-compressed) is safe now that both the merge
        # (writers.las) and read (load_points' explicit readers.las) sides
        # name their reader/writer type outright instead of letting PDAL
        # auto-detect one from the extension -- that auto-detection was the
        # actual bug (a bare .laz filename got treated as COPC by default,
        # and this plain writers.las output has no COPC octree structure).
        lasfile = Path(f'laz-files/{item_id}.laz')
        collection = 'laz-phase2'
        stac = 'https://drwgni8q1h.execute-api.us-west-2.amazonaws.com'

        # Resume support: skip tiles whose full output set is already on S3
        # (a previous run's laz, buildings, and features files), so
        # re-running main.py over the same df doesn't redo completed work.
        laz_key = f"phase2/laz/{item_id}.laz"
        buildings_key = f"phase2/buildings/{item_id}_buffered30m.parquet"
        features_key = f"phase2/features/{item_id}.parquet"
        existing = [k for k in (laz_key, buildings_key, features_key) if s3_object_exists(S3_BUCKET, k)]
        if len(existing) == 3:
            print(f"{existing} already exist on s3, skipping")
            return item_id, "skipped", ""

        # stac call
        item = cbt.get_stac_item(item_id, collection, stac)
        bbox = item.bbox
        print(f'bbox {bbox}')

        # increase bbox by 30 meter buffer
        buffer = 30
        distance = cbt.get_distance_degrees(buffer)
        bbox_buffer = cbt.get_buffered_bbox(bbox, distance)
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

        # building footprints file -- tile_buildings is the exact path
        # get_buffered_tile_footprints already wrote, so reuse it directly
        # instead of reconstructing it (item.parquet isn't a real attribute).
        footprints_file = tile_buildings
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

        # footprints_file is already uploaded to S3 inside
        # get_buffered_tile_footprints and is only needed locally up to this
        # point -- without this it accumulates on the worker's disk forever,
        # once per tile.
        Path(footprints_file).unlink(missing_ok=True)
        print(f'"{footprints_file}" deleted.')

        # export features to parquet
        cpf.export_features(xyz, points, hag, geom_vec, features_file)
        print(f'Delete features file "{features_file}"')
        Path(features_file).unlink(missing_ok=True)
        print(f'"{features_file}" deleted.')

        # lasfile is already uploaded to S3 inside crop_copc and is only
        # needed locally up to this point (feature computation just read
        # it) -- without this it accumulates on the worker's disk forever,
        # once per tile.
        Path(lasfile).unlink(missing_ok=True)
        print(f'"{lasfile}" deleted.')

        return item_id, "ok", ""
    except Exception as e:
        print(f'ERROR processing {item_id}: {e}')
        return item_id, "error", str(e)


############################################################
#                         RUN LOOP                        #
############################################################

# Sequential, one tile at a time -- no client.submit/futures, no Dask. See
# the module-level comment for why (memory, not speed). process() already
# catches its own exceptions and returns an "error" row rather than
# raising, so this try/except is just a backstop for anything that
# genuinely escapes it (e.g. a KeyboardInterrupt mid-tile shouldn't lose
# every result gathered so far).
results = []
for _, row in df.iterrows():
    item_id = row['id']
    try:
        results.append(process(item_id))
    except Exception as e:
        print(f'ERROR processing {item_id}: {e}')
        results.append((item_id, "error", str(e)))

results_df = pd.DataFrame(results, columns=["item_id", "status", "error"])
local_path = Path(tempfile.gettempdir()) / "future_results_local.parquet"
results_df.to_parquet(local_path)
# _local suffix: keeps this run's summary from clobbering the Coiled run's
# phase2/future_results.parquet, since both may be active at once.
upload_to_s3(local_path, S3_BUCKET, "phase2/future_results_local.parquet")
local_path.unlink()
