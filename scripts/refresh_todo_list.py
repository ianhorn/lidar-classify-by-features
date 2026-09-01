#!/usr/bin/env python3

############################################################
# Rebuild stac_item_list.parquet (the Coiled run's todo list)
# from stac_item_list_full.parquet, subtracting:
#   - tiles whose full output set (laz + buildings + features)
#     already exists on S3 -- mirrors main.py's own per-item
#     resume check, done here up front so the submitted df is
#     small instead of relying on main.py to skip 28k rows.
#   - tiles with a laz-problematic/{id}_problematic.parquet
#     manifest (a previous run couldn't crop one of their
#     source hrefs), so known-bad tiles aren't resubmitted to
#     fail again.
# Ian Horn
# September 1, 2026
############################################################

from pathlib import Path

import boto3
import pandas as pd

S3_BUCKET = "lidar-classification"


def list_keys(bucket, prefix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def main():
    repo_root = Path(__file__).resolve().parent.parent
    full_path = repo_root / "stac_item_list_full.parquet"
    output_path = repo_root / "stac_item_list.parquet"

    full_df = pd.read_parquet(full_path)
    print(f"{len(full_df):,} items in full list")

    laz_ids = {Path(k).stem for k in list_keys(S3_BUCKET, "phase2/laz/")}
    buildings_ids = {Path(k).name.removesuffix("_buffered30m.parquet")
                      for k in list_keys(S3_BUCKET, "phase2/buildings/")}
    features_ids = {Path(k).stem for k in list_keys(S3_BUCKET, "phase2/features/")}
    processed_ids = laz_ids & buildings_ids & features_ids
    print(f"{len(laz_ids):,} laz / {len(buildings_ids):,} buildings / "
          f"{len(features_ids):,} features on s3 -> {len(processed_ids):,} tiles with all three")

    problematic_ids = {
        Path(k).name.removesuffix("_problematic.parquet")
        for k in list_keys(S3_BUCKET, "phase2/laz-problematic/")
        if k.endswith("_problematic.parquet")
    }
    print(f"{len(problematic_ids):,} tiles flagged problematic")

    todo_df = full_df[~full_df["id"].isin(processed_ids | problematic_ids)].reset_index(drop=True)
    print(f"{len(todo_df):,} remaining after excluding processed + laz-problematic/")

    todo_df.to_parquet(output_path, compression="zstd")
    print(f"wrote {len(todo_df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
