#!/usr/bin/env python3

"""
Backfill CanopyHeight (see canopy_features.py) into feature tiles' parquets
on S3, from Meta/WRI's canopy height raster.

Like backfill_bridge_features.py, this reads each tile's X/Y, samples the
feature, and rewrites the whole parquet with the new column added -- there's
no way to append a column to an existing parquet object. Every tile gets the
column, including tiles entirely outside the raster's valid area (CanopyHeight
all NaN there), so the schema stays uniform across tiles.

This doesn't change FEATURE_COLUMNS or calculate_point_features.py; adding
the column to the model and retraining is a separate step. A tile whose
parquet already has CanopyHeight is skipped unless --overwrite is passed.

--local tests one parquet on disk without touching S3: it writes
<name>_with_canopy.parquet next to the input.

--raster must point at the Meta/WRI FileGDB raster on the machine running
this (canopy_features.py's RASTER_PATH by default); an AWS Batch run would
need it baked into the image or pulled from S3 first -- not yet done.
"""

import argparse
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.fs
import pyarrow.parquet as pq

from canopy_features import RASTER_PATH, compute_canopy_height
from common import S3_BUCKET, add_keys_file_arg, add_shard_args, list_feature_tiles, restrict_keys, shard_keys, tile_id_from_key, upload_to_s3

DEFAULT_WORKERS = 3  # each in-flight tile's parquet can run multiple GB -- see train_building_classifier.py


def add_canopy_column(source, raster_path, filesystem=None):
    """Read one parquet (S3 path or local) and return it as a DataFrame with CanopyHeight added."""

    xy = pq.read_table(source, columns=["X", "Y"], filesystem=filesystem)
    x, y = xy["X"].to_numpy(), xy["Y"].to_numpy()

    df = pq.read_table(source, filesystem=filesystem).to_pandas()
    df["CanopyHeight"] = compute_canopy_height(x, y, raster_path=raster_path)
    return df


def backfill_tile(key, raster_path, overwrite, s3_fs):
    tile_id = tile_id_from_key(key)
    source = f"{S3_BUCKET}/{key}"

    if not overwrite and "CanopyHeight" in pq.read_schema(source, filesystem=s3_fs).names:
        return tile_id, "skipped", None

    df = add_canopy_column(source, raster_path, filesystem=s3_fs)
    valid_fraction = float(df["CanopyHeight"].notna().mean())

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path)
    upload_to_s3(tmp_path, S3_BUCKET, key)
    tmp_path.unlink()

    return tile_id, "ok", valid_fraction


def run_local(path, raster_path):
    df = add_canopy_column(path, raster_path)
    out = Path(path).with_name(Path(path).stem + "_with_canopy.parquet")
    df.to_parquet(out)
    valid = df["CanopyHeight"].notna().mean()
    print(f"{len(df):,} rows, {valid:.2%} with valid canopy data -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raster", default=RASTER_PATH, help="path to the Meta/WRI canopy height FileGDB raster")
    parser.add_argument("--local", help="process one local features parquet instead of S3 (no upload)")
    parser.add_argument("--n-tiles", type=int, default=None, help="backfill only the first N tiles (for testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true", help="reprocess tiles that already have CanopyHeight")
    add_shard_args(parser)
    add_keys_file_arg(parser)
    args = parser.parse_args()

    if args.local:
        run_local(args.local, args.raster)
        return

    print("listing feature tiles in S3...")
    keys = restrict_keys(list_feature_tiles(), args.keys_file)
    print(f"{len(keys):,} feature tiles available")

    if args.n_tiles is not None:
        keys = keys[: args.n_tiles]
        print(f"limiting to first {len(keys)} tiles for this run")

    keys = shard_keys(keys, args.shard_index, args.num_shards)
    if args.shard_index is not None:
        print(f"shard {args.shard_index}/{args.num_shards}: {len(keys)} tiles")

    s3_fs = pyarrow.fs.S3FileSystem()

    start = time.perf_counter()
    ok = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(backfill_tile, key, args.raster, args.overwrite, s3_fs): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            tile_id = tile_id_from_key(key)
            try:
                tile_id, status, detail = future.result()
            except Exception as e:
                print(f"ERROR backfilling {key}: {e}")
                errors += 1
                continue
            if status == "ok":
                ok += 1
                print(f"{tile_id}: ok ({detail:.2%} valid canopy coverage)")
            else:
                skipped += 1
                print(f"{tile_id}: skipped (already has CanopyHeight)")

    elapsed = time.perf_counter() - start
    print(f"\nprocessed {len(keys)} tiles in {elapsed:.1f}s -- ok={ok} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
