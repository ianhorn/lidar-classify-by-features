#!/usr/bin/env python3

"""
One-off fix for the 35 tiles that failed during the canopy_backfill-v2-window
highmem run with [Errno 28] No space left on device (Fargate ephemeral
storage filled up under --workers 3 concurrency on this run). Not meant to
be reused -- scripts/backfill_canopy_features.py is the real tool; this just
replays the same backfill_tile() call for the specific tiles that didn't
make it, since there's no --tile-keys-file option on that script.
"""

import pyarrow.fs

from backfill_canopy_features import backfill_tile

FAILED_KEYS = ['phase2/features/N150E343_LAS_Phase2.copc.parquet', 'phase2/features/N151E220_LAS_Phase2.copc.parquet', 'phase2/features/N149E407_LAS_Phase2.copc.parquet', 'phase2/features/N150E225_LAS_Phase2.copc.parquet', 'phase2/features/N149E408_LAS_Phase2.copc.parquet', 'phase2/features/N149E348_LAS_Phase2.copc.parquet', 'phase2/features/N150E285_LAS_Phase2.copc.parquet', 'phase2/features/N150E405_LAS_Phase2.copc.parquet', 'phase2/features/N149E410_LAS_Phase2.copc.parquet', 'phase2/features/N151E164_LAS_Phase2.copc.parquet', 'phase2/features/N149E411_LAS_Phase2.copc.parquet', 'phase2/features/N149E351_LAS_Phase2.copc.parquet', 'phase2/features/N151E167_LAS_Phase2.copc.parquet', 'phase2/features/N151E287_LAS_Phase2.copc.parquet', 'phase2/features/N150E350_LAS_Phase2.copc.parquet', 'phase2/features/N150E352_LAS_Phase2.copc.parquet', 'phase2/features/N150E412_LAS_Phase2.copc.parquet', 'phase2/features/N150E113_LAS_Phase2.copc.parquet', 'phase2/features/N149E416_LAS_Phase2.copc.parquet', 'phase2/features/N107E423_LAS_Phase2.copc.parquet', 'phase2/features/N150E185_LAS_Phase2.copc.parquet', 'phase2/features/N149E368_LAS_Phase2.copc.parquet', 'phase2/features/N149E371_LAS_Phase2.copc.parquet', 'phase2/features/N150E132_LAS_Phase2.copc.parquet', 'phase2/features/N149E315_LAS_Phase2.copc.parquet', 'phase2/features/N150E252_LAS_Phase2.copc.parquet', 'phase2/features/N149E319_LAS_Phase2.copc.parquet', 'phase2/features/N149E380_LAS_Phase2.copc.parquet', 'phase2/features/N110E359_LAS_Phase2.copc.parquet', 'phase2/features/N150E155_LAS_Phase2.copc.parquet', 'phase2/features/N150E215_LAS_Phase2.copc.parquet', 'phase2/features/N150E275_LAS_Phase2.copc.parquet', 'phase2/features/N111E390_LAS_Phase2.copc.parquet', 'phase2/features/N149E399_LAS_Phase2.copc.parquet', 'phase2/features/N149E402_LAS_Phase2.copc.parquet']

RASTER_PATH = "/vsis3/lidar-classification/canopy/ky_canopy_height_3857.tif"


def main():
    s3_fs = pyarrow.fs.S3FileSystem()
    ok = errors = 0
    for key in FAILED_KEYS:
        try:
            tile_id, status, detail = backfill_tile(key, RASTER_PATH, True, s3_fs)
            print(f"{tile_id}: {status} ({detail:.2%} valid canopy coverage)")
            ok += 1
        except Exception as e:
            print(f"ERROR backfilling {key}: {e}")
            errors += 1
    print(f"done: ok={ok} errors={errors}")


if __name__ == "__main__":
    main()
