#!/usr/bin/env python3

"""
Backfill DistToBridge/InBridgeBuffer (see bridge_features.py) into feature
tiles' parquets on S3.

Both columns come from each point's X/Y and the KYTC bridge lines, so this
never touches the .laz files -- it reads X/Y, computes the two columns, and
rewrites the parquet with them added (there's no way to append a column to an
existing parquet object, same as backfill_return_number.py). Every tile gets
both columns, including tiles with no bridges nearby (DistToBridge at its cap,
InBridgeBuffer False), so the schema stays uniform across tiles.

Like backfill_return_number.py this doesn't change FEATURE_COLUMNS or
calculate_point_features.py; adding the columns to the model and retraining is
a separate step. A tile whose parquet already has InBridgeBuffer is skipped
unless --overwrite is passed, so a re-run doesn't redo finished work.

--local tests one parquet on disk without touching S3: it writes
<name>_with_bridges.parquet next to the input.

--bridges must point at BridgeLocationsLine.shp on the machine running this;
an AWS Batch run would need the shapefile baked into the image or pulled from
S3 first.

--checkpoint PATH adds a *local* resume log on top of the S3-based skip
check above -- every tile_id this run confirms (ok or skipped) gets appended
to it immediately, and on startup the file's contents are read back and
excluded from the run before anything else happens. The S3 check alone is
already resume-safe (nothing on S3 gets redone), but re-scanning all ~45k
tiles' schemas from the top on every restart is real time on an unstable
host -- this skips that re-scan too. Written for the post-race-condition
cleanup pass, which kept getting killed by unrelated host restarts.
"""

import argparse
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.fs
import pyarrow.parquet as pq

from bridge_features import LINES_SHP, MAX_DIST_FT, bridges_near_points, compute_bridge_features, load_bridges
from common import S3_BUCKET, add_keys_file_arg, add_shard_args, list_feature_tiles, restrict_keys, shard_keys, tile_id_from_key, upload_to_s3

DEFAULT_WORKERS = 3  # each in-flight tile's parquet can run multiple GB -- see train_building_classifier.py


def add_bridge_columns(source, bridges, filesystem=None):
    """Read one parquet (S3 path or local) and return it as a DataFrame with the bridge columns added."""

    xy = pq.read_table(source, columns=["X", "Y"], filesystem=filesystem)
    x, y = xy["X"].to_numpy(), xy["Y"].to_numpy()

    nearby = bridges_near_points(bridges, x, y)
    if len(nearby):
        dist, in_buffer = compute_bridge_features(x, y, nearby)
    else:
        dist = np.full(len(x), MAX_DIST_FT, dtype=np.float32)
        in_buffer = np.zeros(len(x), dtype=bool)

    df = pq.read_table(source, filesystem=filesystem).to_pandas()
    df["DistToBridge"] = dist
    df["InBridgeBuffer"] = in_buffer
    return df, len(nearby)


def backfill_tile(key, bridges, overwrite, s3_fs):
    tile_id = tile_id_from_key(key)
    source = f"{S3_BUCKET}/{key}"

    if not overwrite and "InBridgeBuffer" in pq.read_schema(source, filesystem=s3_fs).names:
        return tile_id, "skipped", None

    df, n_bridges = add_bridge_columns(source, bridges, filesystem=s3_fs)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path)
    upload_to_s3(tmp_path, S3_BUCKET, key)
    tmp_path.unlink()

    return tile_id, "ok", n_bridges


def run_local(path, bridges):
    df, n_bridges = add_bridge_columns(path, bridges)
    out = Path(path).with_name(Path(path).stem + "_with_bridges.parquet")
    df.to_parquet(out)
    print(f"{len(df):,} rows, {n_bridges} nearby bridges, {df['InBridgeBuffer'].sum():,} in buffer -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bridges", default=LINES_SHP, help="path to BridgeLocationsLine.shp")
    parser.add_argument("--local", help="process one local features parquet instead of S3 (no upload)")
    parser.add_argument("--n-tiles", type=int, default=None, help="backfill only the first N tiles (for testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true", help="reprocess tiles that already have InBridgeBuffer")
    parser.add_argument("--checkpoint", help="local file to record/resume confirmed tile_ids across restarts")
    add_shard_args(parser)
    add_keys_file_arg(parser)
    args = parser.parse_args()

    bridges = load_bridges(args.bridges)
    print(f"{len(bridges):,} on-structure, non-culvert bridge records")

    if args.local:
        run_local(args.local, bridges)
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

    checkpoint_file = None
    already_confirmed = set()
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            already_confirmed = set(checkpoint_path.read_text().split())
            print(f"checkpoint: {len(already_confirmed):,} tile_ids already confirmed, skipping without an S3 check")
        before = len(keys)
        keys = [key for key in keys if tile_id_from_key(key) not in already_confirmed]
        print(f"checkpoint: {before - len(keys):,} tiles skipped locally, {len(keys):,} remaining")
        checkpoint_file = checkpoint_path.open("a")
    checkpoint_lock = threading.Lock()

    s3_fs = pyarrow.fs.S3FileSystem()

    start = time.perf_counter()
    ok = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(backfill_tile, key, bridges, args.overwrite, s3_fs): key for key in keys}
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
                print(f"{tile_id}: ok ({detail} nearby bridges)")
            else:
                skipped += 1
                print(f"{tile_id}: skipped (already has InBridgeBuffer)")
            if checkpoint_file:
                with checkpoint_lock:
                    checkpoint_file.write(tile_id + "\n")
                    checkpoint_file.flush()

    if checkpoint_file:
        checkpoint_file.close()

    elapsed = time.perf_counter() - start
    print(f"\nprocessed {len(keys)} tiles in {elapsed:.1f}s -- ok={ok} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
