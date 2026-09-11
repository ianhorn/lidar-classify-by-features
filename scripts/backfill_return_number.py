#!/usr/bin/env python3

"""
Backfill ReturnNumber/NumberOfReturns into feature tiles that were computed
before those columns existed in the schema.

Unlike everything else in calculate_point_features.py, these two columns
don't need filters.hag_delaunay or the PCA/KDTree geometric-feature pass --
they're raw per-point fields already sitting in every source .laz (LAS point
formats 0-10 all carry them), so this reads them straight off a presigned S3
URL with a bare readers.las pipeline and never downloads a full .laz to
disk. That's the whole reason this is a separate backfill script instead of
just re-running calculate_point_features.py: it skips the expensive part.

Row order between a tile's .laz and its features parquet is positional, not
keyed -- both calculate_point_features.py's load_points() and this script's
raw read go through PDAL's readers.las with no intervening sort/reorder
step, so parquet row i is laz point i. This is verified per-tile (row counts
must match) before anything is written; a mismatch skips the tile with an
error rather than silently misaligning columns.

This only touches feature parquets that already exist on S3 -- it doesn't
change calculate_point_features.py or FEATURE_COLUMNS, and it doesn't touch
tiles phase 1 hasn't gotten to yet. That's deliberate: phase 1 is still
running feature extraction with the old schema on other machines, and this
script doesn't interact with that pipeline at all. Deciding whether/when to
add ReturnNumber/NumberOfReturns to FEATURE_COLUMNS and retrain is a
separate step, not part of this script.

Resume support: a tile whose features parquet already has a ReturnNumber
column is skipped unless --overwrite is passed, so a re-run over the same
tile list doesn't redo completed work.

--shard-index/--num-shards (see common.py's add_shard_args/shard_keys) split
the tile list into disjoint chunks for running this as an AWS Batch array
job -- see docker/Dockerfile. --shard-index defaults to
$AWS_BATCH_JOB_ARRAY_INDEX so a job definition's command doesn't need to
interpolate it.
"""

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
import pdal

from common import S3_BUCKET, add_shard_args, list_feature_tiles, shard_keys, tile_id_from_key, upload_to_s3

LAZ_PREFIX = "phase2/laz/"
DEFAULT_WORKERS = 3  # each in-flight tile's parquet can run multiple GB -- see train_building_classifier.py


def read_return_fields(tile_id):
    """Read ReturnNumber/NumberOfReturns for one tile straight off S3, no local download."""

    s3 = boto3.client("s3")
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": f"{LAZ_PREFIX}{tile_id}.laz"}, ExpiresIn=900
    )
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.las", "filename": url}]}))
    pipeline.execute()
    arr = pipeline.arrays[0]
    return arr["ReturnNumber"], arr["NumberOfReturns"]


def backfill_tile(key, overwrite):
    """
    Add ReturnNumber/NumberOfReturns to one tile's features parquet, in place
    (same S3 key). Reads and rewrites the whole parquet -- there's no way to
    append a column to an existing parquet object without doing that.
    """

    tile_id = tile_id_from_key(key)
    path = f"s3://{S3_BUCKET}/{key}"

    df = pd.read_parquet(path)
    if not overwrite and "ReturnNumber" in df.columns:
        return tile_id, "skipped", None

    return_number, number_of_returns = read_return_fields(tile_id)
    if len(return_number) != len(df):
        return tile_id, "error", f"point count mismatch: laz={len(return_number):,} parquet={len(df):,}"

    df["ReturnNumber"] = return_number
    df["NumberOfReturns"] = number_of_returns

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path)
    upload_to_s3(tmp_path, S3_BUCKET, key)
    tmp_path.unlink()

    return tile_id, "ok", None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tiles", type=int, default=None, help="backfill only the first N tiles (for testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true", help="reprocess tiles that already have ReturnNumber")
    add_shard_args(parser)
    args = parser.parse_args()

    print("listing feature tiles in S3...")
    keys = list_feature_tiles()
    print(f"{len(keys):,} feature tiles available")

    if args.n_tiles is not None:
        keys = keys[: args.n_tiles]
        print(f"limiting to first {len(keys)} tiles for this run")

    keys = shard_keys(keys, args.shard_index, args.num_shards)
    if args.shard_index is not None:
        print(f"shard {args.shard_index}/{args.num_shards}: {len(keys)} tiles")

    start = time.perf_counter()
    ok = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(backfill_tile, key, args.overwrite): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            tile_id = tile_id_from_key(key)
            try:
                tile_id, status, error = future.result()
            except Exception as e:
                print(f"ERROR backfilling {key}: {e}")
                errors += 1
                continue
            if status == "ok":
                ok += 1
                print(f"{tile_id}: ok")
            elif status == "skipped":
                skipped += 1
                print(f"{tile_id}: skipped (already has ReturnNumber)")
            else:
                errors += 1
                print(f"{tile_id}: ERROR ({error})")

    elapsed = time.perf_counter() - start
    print(f"\nprocessed {len(keys)} tiles in {elapsed:.1f}s -- ok={ok} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
