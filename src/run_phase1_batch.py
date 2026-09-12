#!/usr/bin/env python3

"""
Phase 1 (feature extraction) driver for AWS Batch, as an alternative to
main.py's Coiled-cluster version -- built because the Coiled organization
ran out of compute credits (renews 2026-10-01) with ~2,922 tiles still
remaining. This is the same per-tile pipeline as main.py's process()
(STAC search -> crop COPC -> Overture footprint buffer -> geometric
features -> upload laz/buildings/features to S3), just restructured to run
as a Batch array job instead of Dask tasks on a Coiled cluster.

One tile at a time per task, not a thread pool across tiles -- matching
main.py's own worker_options={"nthreads": 1}: Coiled's comments there note
concurrent tile processing on one worker blew past its memory budget on
dense tiles (one buffered tile alone had ~15.8M points), even after
bumping worker_memory. Batch's own array-job parallelism (many tasks, not
many threads within one task) is where the real parallelism comes from
here instead.

--shard-index/--num-shards split the tile list into disjoint chunks, same
pattern as scripts/common.py's add_shard_args/shard_keys (duplicated here
rather than imported across the src/ vs scripts/ directory split, to keep
this script self-contained inside the Docker image).
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor

import calculate_point_features as cpf
import create_buffered_tile as cbt
import numpy as np
from scipy.spatial import cKDTree

S3_BUCKET = "lidar-classification"
STAC_URL = "https://drwgni8q1h.execute-api.us-west-2.amazonaws.com"
COLLECTION = "laz-phase2"
OVERTURE_RELEASE = "2026-07-22.0"
OVERTURE_URL = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*"


def s3_object_exists(bucket, key):
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def upload_to_s3(local_path, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded {local_path} to s3://{bucket}/{key}")


def shard_ids(ids, shard_index, num_shards):
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be given together")
    if shard_index is None:
        return ids
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard-index {shard_index} out of range for --num-shards {num_shards}")
    return sorted(ids)[shard_index::num_shards]


def process_tile(item_id):
    """
    Same pipeline as main.py's process(): STAC search, crop COPC, buffer
    Overture footprints, compute geometric features, upload laz/buildings/
    features to S3. Copied rather than imported from main.py because
    main.py has module-level side effects (creates a Coiled cluster on
    import) that don't apply here.
    """

    output_id = item_id
    if "copz" in output_id:
        output_id = output_id.replace("copz", "copc")

    lasfile = Path(f"laz-files/{output_id}.laz")

    laz_key = f"phase2/laz/{output_id}.laz"
    buildings_key = f"phase2/buildings/{output_id}_buffered30m.parquet"
    features_key = f"phase2/features/{output_id}.parquet"
    existing = [k for k in (laz_key, buildings_key, features_key) if s3_object_exists(S3_BUCKET, k)]
    if len(existing) == 3:
        return output_id, "skipped", ""

    # main.py's blind copz->copc replacement before the STAC fetch 404s for
    # some tiles -- the catalog has them registered under their original
    # .copz id even though every other item (and our own S3 naming) uses
    # .copc. Confirmed directly: N073E353_LAS_Phase2.copc 404s,
    # N073E353_LAS_Phase2.copz resolves fine. Fetch with whatever id the
    # catalog actually has, then normalize item.id itself so every
    # downstream consumer (features_file below, get_buffered_tile_footprints'
    # own output filename) still uses the same .copc-style naming as every
    # other tile, keeping S3 output consistent either way.
    item = cbt.get_stac_item(item_id, COLLECTION, STAC_URL)
    item.id = output_id
    bbox = item.bbox

    buffer = 30
    distance = cbt.get_distance_degrees(buffer)
    bbox_buffer = cbt.get_buffered_bbox(bbox, distance)
    hrefs = cbt.search_stac(STAC_URL, COLLECTION, bbox_buffer)

    bbox_3089 = cbt.reproject_bbox(bbox_buffer)
    bounds = cbt.pdal_bounds(bbox_3089)

    with ThreadPoolExecutor(max_workers=2) as executor:
        crop_future = executor.submit(cbt.crop_copc, hrefs, bounds, lasfile)
        footprints_future = executor.submit(cbt.get_buffered_tile_footprints, item, OVERTURE_URL, bbox_buffer)
        crop_future.result()
        tile_buildings = footprints_future.result()

    footprints_file = tile_buildings
    features_file = f"lidar-features/{item.id}.parquet"
    radius = 7.0
    chunk_size = 90_000

    points = cpf.load_points(lasfile)
    hag = points["HeightAboveGround"]

    xyz = np.column_stack([points["X"], points["Y"], points["Z"]])
    tree = cKDTree(xyz)

    geom_vec = cpf.compute_geometric_features_vectorized(tree, xyz, radius=radius, chunk_size=chunk_size)

    within_footprint, footprint_height = cpf.compute_within_footprint(xyz, footprints_file)
    geom_vec["WithinFootprint"] = within_footprint
    geom_vec["FootprintHeight"] = footprint_height

    Path(footprints_file).unlink(missing_ok=True)

    cpf.export_features(xyz, points, hag, geom_vec, features_file)
    Path(features_file).unlink(missing_ok=True)
    Path(lasfile).unlink(missing_ok=True)

    return output_id, "ok", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tiles", type=int, default=None, help="process only the first N tiles (for testing)")
    parser.add_argument(
        "--shard-index", type=int,
        default=int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"]) if "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ else None,
    )
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()

    for d in ("laz-files", "building-files", "lidar-features"):
        Path(d).mkdir(exist_ok=True)

    df = pd.read_parquet(f"s3://{S3_BUCKET}/phase2/stac_item_list.parquet")
    ids = df["id"].tolist()
    print(f"{len(ids):,} tiles in todo list")

    if args.n_tiles is not None:
        ids = ids[: args.n_tiles]
        print(f"limiting to first {len(ids)} tiles for this run")

    ids = shard_ids(ids, args.shard_index, args.num_shards)
    if args.shard_index is not None:
        print(f"shard {args.shard_index}/{args.num_shards}: {len(ids)} tiles")

    start = time.perf_counter()
    results = []
    for item_id in ids:
        try:
            item_id, status, error = process_tile(item_id)
            print(f"{item_id}: {status}")
        except Exception as e:
            print(f"ERROR processing {item_id}: {e}")
            status, error = "error", str(e)
        results.append((item_id, status, error))

    elapsed = time.perf_counter() - start
    print(f"\nprocessed {len(results)} tiles in {elapsed:.1f}s")

    results_df = pd.DataFrame(results, columns=["item_id", "status", "error"])
    results_key = (
        f"phase2/phase1_results/shard_{args.shard_index}_of_{args.num_shards}.parquet"
        if args.shard_index is not None
        else "phase2/phase1_results.parquet"
    )
    local_path = Path(tempfile.gettempdir()) / "phase1_results.parquet"
    results_df.to_parquet(local_path)
    upload_to_s3(local_path, S3_BUCKET, results_key)
    local_path.unlink()


if __name__ == "__main__":
    main()
