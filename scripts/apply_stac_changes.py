#!/usr/bin/env python3

"""
Apply the output of diff_stac_changes.py. Each step is its own flag so you can
inspect between them; everything destructive is a dry run unless --execute.

  --stage          upload stac_stage/manifest.parquet + bucket_tiles.parquet to
                   s3://lidar-classification/phase2/sync/ (what run_sync_batch.py
                   reads) and write stac_stage/feature_keys.txt for the backfills
  --delete-orphans delete laz/buildings/features/classified outputs + stale
                   problematic/unprocessable flags for tiles that no longer
                   exist in the bucket (stac_stage/orphaned.txt)
  --refresh-full-list
                   rewrite stac_item_list_full.parquet from the bucket tiles
                   (so refresh_todo_list.py works off current reality); with
                   --upload also pushes it to phase2/stac_item_list_full.parquet
"""

import argparse
from pathlib import Path

import boto3
import pandas as pd

BUCKET = "lidar-classification"
REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "stac_stage"
SRC_S3 = "s3://kyfromabove/elevation/PointCloud/Phase2/"


def orphan_keys(oid):
    return [
        f"phase2/laz/{oid}.laz",
        f"phase2/buildings/{oid}_buffered30m.parquet",
        f"phase2/features/{oid}.parquet",
        f"phase2/classified/{oid}.parquet",
        f"phase2/laz-problematic/{oid}_problematic.parquet",
        f"phase2/laz-unprocessable/{oid}_unprocessable.parquet",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--delete-orphans", action="store_true")
    parser.add_argument("--refresh-full-list", action="store_true")
    parser.add_argument("--upload", action="store_true", help="with --refresh-full-list: also upload to S3")
    parser.add_argument("--execute", action="store_true", help="actually delete (default: dry run)")
    args = parser.parse_args()
    s3 = boto3.client("s3")

    if args.stage:
        manifest = pd.read_parquet(STAGE / "manifest.parquet")
        (STAGE / "feature_keys.txt").write_text(
            "\n".join(f"phase2/features/{o}.parquet" for o in manifest["output_id"]) + "\n")
        for name in ("manifest.parquet", "bucket_tiles.parquet"):
            s3.upload_file(str(STAGE / name), BUCKET, f"phase2/sync/{name}")
            print(f"uploaded {name} -> s3://{BUCKET}/phase2/sync/{name}")
        print(f"{len(manifest)} tiles staged ({manifest['action'].value_counts().to_dict()}); "
              f"wrote stac_stage/feature_keys.txt")

    if args.delete_orphans:
        orphans = (STAGE / "orphaned.txt").read_text().split()
        existing = []
        for oid in orphans:
            for key in orphan_keys(oid):
                try:
                    s3.head_object(Bucket=BUCKET, Key=key)
                    existing.append(key)
                except s3.exceptions.ClientError:
                    pass
        print(f"{len(orphans)} orphaned tiles, {len(existing)} objects to delete")
        if args.execute:
            for i in range(0, len(existing), 1000):
                s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in existing[i:i + 1000]]})
            print("deleted")
        else:
            print("dry run -- first few:\n  " + "\n  ".join(existing[:6]) + "\n(pass --execute to delete)")

    if args.refresh_full_list:
        t = pd.read_parquet(STAGE / "bucket_tiles.parquet")
        df = pd.DataFrame({
            "id": t["output_id"],
            "href": t["href"],
            "s3_uri": SRC_S3 + t["filename"],
            "bbox_minx": t["bbox"].str[0], "bbox_miny": t["bbox"].str[1],
            "bbox_maxx": t["bbox"].str[2], "bbox_maxy": t["bbox"].str[3],
        })
        out = REPO / "stac_item_list_full.parquet"
        df.to_parquet(out, compression="zstd")
        print(f"wrote {len(df):,} rows to {out}")
        if args.upload:
            s3.upload_file(str(out), BUCKET, "phase2/stac_item_list_full.parquet")
            print("uploaded to S3")


if __name__ == "__main__":
    main()
