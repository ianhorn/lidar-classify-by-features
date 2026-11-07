#!/usr/bin/env python3

"""
Phase 1 for tiles that changed upstream (new / replaced / retry), driven by
the bucket-derived manifest from scripts/diff_stac_changes.py instead of the
(stale) STAC catalog. Same per-tile pipeline as run_phase1_batch.py -- this
just feeds it bbox + neighbor hrefs from the manifest and forces overwrite.

Reads (uploaded by scripts/apply_stac_changes.py --stage):
  s3://lidar-classification/phase2/sync/manifest.parquet       tiles to process
  s3://lidar-classification/phase2/sync/bucket_tiles.parquet   every bucket tile (for buffer neighbors)

On success a tile's stale laz-problematic/ flag (the "retry" action) is
deleted so refresh_todo_list.py stops excluding it. Per-shard results go to
phase2/sync/results/shard_{i}_of_{n}.parquet.
"""

import argparse
import os
import tempfile
from pathlib import Path

import boto3
import pandas as pd

import run_phase1_batch as p1

SYNC_PREFIX = "phase2/sync"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tiles", type=int, default=None)
    parser.add_argument("--shard-index", type=int,
                        default=int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"]) if "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ else None)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()

    for d in ("laz-files", "building-files", "lidar-features"):
        Path(d).mkdir(exist_ok=True)

    manifest = pd.read_parquet(f"s3://{p1.S3_BUCKET}/{SYNC_PREFIX}/manifest.parquet")
    bucket_tiles = pd.read_parquet(f"s3://{p1.S3_BUCKET}/{SYNC_PREFIX}/bucket_tiles.parquet")
    neighbors = [(tuple(b), h) for b, h in zip(bucket_tiles["bbox"], bucket_tiles["href"])]

    # Resume support: force=True overwrites stale outputs, so "already exists"
    # can't be the skip test. Instead, a tile is done for THIS sync if its
    # features parquet was written after the manifest was staged -- so a Spot-
    # interrupted shard that restarts skips what it already finished.
    s3 = boto3.client("s3")
    staged_at = s3.head_object(Bucket=p1.S3_BUCKET, Key=f"{SYNC_PREFIX}/manifest.parquet")["LastModified"]

    def done_since_staging(oid):
        try:
            head = s3.head_object(Bucket=p1.S3_BUCKET, Key=f"phase2/features/{oid}.parquet")
        except s3.exceptions.ClientError:
            return False
        return head["LastModified"] > staged_at

    by_id = manifest.set_index("output_id")
    ids = list(by_id.index)
    if args.n_tiles is not None:
        ids = ids[: args.n_tiles]
    ids = p1.shard_ids(ids, args.shard_index, args.num_shards)
    print(f"{len(ids)} tiles in this shard ({len(manifest)} in manifest)")

    results = []
    for oid in ids:
        row = by_id.loc[oid]
        if done_since_staging(oid):
            print(f"{oid}: skipped (already done in this sync)")
            results.append((oid, row.action, "skipped", ""))
            continue
        bbox = [row.bbox_minx, row.bbox_miny, row.bbox_maxx, row.bbox_maxy]
        try:
            _, status, error = p1.process_tile(oid, tile=(bbox, neighbors), force=True)
            if row.action == "retry":
                for suffix in ("laz-problematic/{}_problematic.parquet", "laz-unprocessable/{}_unprocessable.parquet"):
                    s3.delete_object(Bucket=p1.S3_BUCKET, Key="phase2/" + suffix.format(oid))
            print(f"{oid}: {status} ({row.action})")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"ERROR processing {oid}: {e}\n{tb}")
            status, error = "error", f"{e}\n{tb}"
            if "can't build a ground TIN" in str(e):
                p1.report_unprocessable(oid, error)
        results.append((oid, row.action, status, error))

    results_df = pd.DataFrame(results, columns=["output_id", "action", "status", "error"])
    key = (f"{SYNC_PREFIX}/results/shard_{args.shard_index}_of_{args.num_shards}.parquet"
           if args.shard_index is not None else f"{SYNC_PREFIX}/results/all.parquet")
    local = Path(tempfile.gettempdir()) / "sync_results.parquet"
    results_df.to_parquet(local)
    p1.upload_to_s3(local, p1.S3_BUCKET, key)
    print(results_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
