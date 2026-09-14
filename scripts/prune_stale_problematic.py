#!/usr/bin/env python3

"""
Delete laz-problematic/ manifests for tiles that actually completed
successfully anyway.

report_problematic() (in create_buffered_tile.py) writes a manifest
unconditionally whenever crop_copc() hits even one failed href, regardless
of whether the tile ultimately succeeds overall (crop_copc only hard-fails
if *every* href fails) -- and nothing ever cleans that manifest up
afterward, even once the tile completes. So laz-problematic/ accumulates a
permanent record mixing tiles that are genuinely stuck (every href failed)
with tiles that just had a one-off blip and finished fine anyway.

This only deletes manifests for tiles that are fully processed (laz +
buildings + features all present on S3) -- manifests for tiles that never
completed are left alone, since those are the genuinely-still-broken ones
refresh_todo_list.py correctly excludes from re-attempts.
"""

import boto3

S3_BUCKET = "lidar-classification"


def list_ids(bucket, prefix, suffix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(suffix):
                keys[k.rsplit("/", 1)[-1][: -len(suffix)]] = k
    return keys


def main():
    laz = set(list_ids(S3_BUCKET, "phase2/laz/", ".laz"))
    buildings = {i.removesuffix("_buffered30m") for i in list_ids(S3_BUCKET, "phase2/buildings/", ".parquet")}
    features = set(list_ids(S3_BUCKET, "phase2/features/", ".parquet"))
    processed = laz & buildings & features
    print(f"{len(processed):,} tiles fully processed")

    problematic = list_ids(S3_BUCKET, "phase2/laz-problematic/", "_problematic.parquet")
    print(f"{len(problematic):,} problematic manifests")

    stale_keys = [key for tile_id, key in problematic.items() if tile_id in processed]
    print(f"{len(stale_keys):,} manifests are stale (tile completed successfully anyway) -- deleting")

    s3 = boto3.client("s3")
    for i in range(0, len(stale_keys), 1000):
        batch = stale_keys[i : i + 1000]
        resp = s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True})
        errors = resp.get("Errors", [])
        if errors:
            print(f"ERRORS deleting batch {i}: {errors}")
        print(f"deleted {i + len(batch)}/{len(stale_keys)}")

    print(f"done -- {len(problematic) - len(stale_keys):,} manifests remain (genuinely unprocessed tiles)")


if __name__ == "__main__":
    main()
