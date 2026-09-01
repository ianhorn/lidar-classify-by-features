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
df = pd.read_parquet('s3://lidar-classification/phase2/stac_item_list.parquet')
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
#                    CREATE A CLUSTER                      #
############################################################

cluster = Cluster(
    software='lidar-classification',
    # 150 exceeded the AWS account's current vCPU quota for the m6i.2xlarge
    # bucket (698 vCPU limit / 8 vCPU per worker -> ~85 max); Coiled only
    # provisioned 86 of 150 and the rest errored out. Staying under that
    # ceiling with margin.
    n_workers=80,
    # Default no_client_timeout is 2 minutes -- if the local client
    # disconnects (e.g. this machine sleeps) for longer than that, Coiled
    # shuts down the whole cluster, killing every task in flight. This is a
    # many-hour job across 45k tiles, so give it a lot of headroom.
    no_client_timeout="12 hours",
    # Hard cap on this session regardless of activity -- unlike
    # no_client_timeout (which only fires after the client disconnects),
    # this shuts the cluster down at 4.5 hours even while it's actively
    # working, so a run can't be forgotten and left billing indefinitely.
    cluster_timeout="4.5 hours",
    # Workers were getting killed by Dask's own memory manager mid-tile
    # (hit 80% of the default m6i.xlarge's 16GiB during feature computation
    # on dense tiles, e.g. one buffered tile alone had ~15.8M points) --
    # every task in flight at that moment silently gets lost, with no error
    # message, blocking every tile from ever reaching the features upload.
    # Bumping worker_memory alone didn't fix it: Coiled picked a
    # proportionally bigger/more-CPU instance (8 vCPU for 32GiB), and Dask
    # defaults to one task per thread -- so 8 tiles' feature computations
    # ran concurrently on one worker and blew the budget right back to 80%
    # anyway. Capping threads bounds how many tiles' features get computed
    # at once per worker, independent of instance size.
    worker_memory="32GiB",
    worker_options={"nthreads": 1},
    # Workers hit a 100%-repro UnicodeDecodeError inside pyproj's PROJ log
    # callback on every WGS84->NAD83 transform (byte 0x80 at the same offset
    # regardless of the bbox), which corrupts the transform into returning
    # inf instead of raising cleanly. That matches PROJ emitting a non-UTF8
    # warning (commonly a degree symbol) when no locale is set, which the
    # miniforge base image doesn't set by default -- force one.
    environ={"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
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

futures = {}
for _, row in df.iterrows():
    future = client.submit(process, row['id'])
    futures[future] = row['id']

# client.gather(futures) raises on the first cancelled/errored future and
# discards every other result along with it -- a single dead worker would
# lose the whole batch's output. Gather each future individually instead so
# one bad task only costs its own row.
results = []
for future, item_id in futures.items():
    try:
        results.append(future.result())
    except Exception as e:
        print(f'ERROR gathering {item_id}: {e}')
        results.append((item_id, "error", str(e)))

results_df = pd.DataFrame(results, columns=["item_id", "status", "error"])
local_path = Path(tempfile.gettempdir()) / "future_results.parquet"
results_df.to_parquet(local_path)
upload_to_s3(local_path, S3_BUCKET, "phase2/future_results.parquet")
local_path.unlink()
