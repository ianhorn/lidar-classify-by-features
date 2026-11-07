#!/usr/bin/env python3

"""
Shared constants/helpers for the building-classification scripts.
FEATURE_COLUMNS must stay identical between training and inference -- not
because order matters (XGBoost's sklearn API matches by column name, not
position), but because the *set* must match exactly, or predict() raises
on an unseen/missing column.
"""

import os

import boto3
from botocore.exceptions import ClientError

S3_BUCKET = "lidar-classification"
FEATURES_PREFIX = "phase2/features/"

# Geometry/reflectance columns from calculate_point_features.py. X/Y/Z are
# left out (a model trained on absolute coordinates overfits to specific
# tile locations instead of learning geometry that generalizes), as is the
# existing Classification field (that's the LAS source's own noise/ground
# tagging, which is suspected to be wrong in places -- this model should
# judge each point on its own geometry, not defer to a flag that might be
# incorrect) and WithinFootprint (that's the label, not a feature).
#
# ReturnNumber/NumberOfReturns (backfilled across all feature tiles as of
# 2026-09-13) were added specifically to help separate tree canopy from
# rooftops -- tile N142E427 showed canopy false positives sitting at
# building-like HAG (median 11.8ft) and overlapping real-building Intensity
# (35-45K range), which neither HAG nor Intensity alone can filter. Canopy
# is semi-transparent to lidar (a pulse often returns multiple times off
# leaves/branches at different heights before, if ever, reaching a lower
# surface), while a roof is a single hard reflective surface -- almost every
# roof-strike pulse returns once. ReturnNumber > 1 or NumberOfReturns > 1
# should read as "probably not a hard rooftop surface" in a way HAG/Intensity
# can't capture.
#
# DistToBridge/InBridgeBuffer (see bridge_features.py, backfilled 2026-10-13)
# address a different false-positive source: bridge decks, which sit at
# building-like HAG and are flat/reflective like a roof, so neither of the
# canopy-oriented features above catches them. They're computed from the
# KYTC bridge inventory, independent of the point's own geometry.
#
# CanopyHeight (see canopy_features.py, backfilled 2026-10-13) is a second,
# independent canopy signal from Meta/WRI's canopy height raster -- unlike
# ReturnNumber/NumberOfReturns, it doesn't depend on how a given lidar pulse
# happened to return, so it should be more robust on tiles where that proxy
# is weaker. NaN (no raster data at that point, e.g. outside Kentucky) is
# left as-is rather than filled -- XGBoost treats NaN as a native missing
# value, which is what it actually is here, not a meaningful default.
FEATURE_COLUMNS = [
    "HAG",
    "Planarity",
    "Linearity",
    "Sphericity",
    "SurfaceVariation",
    "Roughness",
    "Verticality",
    "NormalX",
    "NormalY",
    "NormalZ",
    "NeighborCount",
    "Density",
    "Intensity",
    "ReturnNumber",
    "NumberOfReturns",
    "DistToBridge",
    "InBridgeBuffer",
    "CanopyHeight",
]
LABEL_COLUMN = "WithinFootprint"


def list_feature_tiles(bucket=S3_BUCKET, prefix=FEATURES_PREFIX):
    """List every features parquet already available in S3."""

    s3 = boto3.client("s3")
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return keys


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
    """Upload a local file to S3."""

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded {local_path} to s3://{bucket}/{key}")


def tile_id_from_key(key):
    return key.rsplit("/", 1)[-1].removesuffix(".parquet")


def add_shard_args(parser):
    """
    --shard-index/--num-shards split a tile-key list into disjoint chunks --
    the pattern an AWS Batch array job needs, where each array child
    processes one shard instead of the whole list in one process.
    shard-index defaults to $AWS_BATCH_JOB_ARRAY_INDEX (which Batch sets
    automatically in every array-job container) so a job definition's
    command doesn't need to interpolate it manually. num-shards still has
    to be passed explicitly -- Batch doesn't expose a child's own array
    size to it, only whoever submits the array job knows that.
    """

    parser.add_argument(
        "--shard-index", type=int,
        default=int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"]) if "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ else None,
        help="which shard this run processes (0-based); defaults to $AWS_BATCH_JOB_ARRAY_INDEX if set",
    )
    parser.add_argument("--num-shards", type=int, default=None,
                         help="total number of shards (must match the Batch array size)")


def shard_keys(keys, shard_index, num_shards):
    """
    Split keys into num_shards disjoint, deterministic pieces and return
    shard_index's piece (or all of keys if sharding wasn't requested).
    Round-robin (keys[i::n]) rather than contiguous blocks so shards land
    roughly balanced even though key order correlates with geography (tile
    id), not file size -- a contiguous block could otherwise land a run of
    unusually large/small tiles all in one shard. Sorted explicitly here
    rather than trusting caller order, so sharding is deterministic even if
    S3 listing order ever isn't.
    """

    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be given together")
    if shard_index is None:
        return keys
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard-index {shard_index} out of range for --num-shards {num_shards}")
    return sorted(keys)[shard_index::num_shards]


def add_keys_file_arg(parser):
    parser.add_argument("--keys-file", help="text file of tile ids or feature keys (one per line); "
                                            "restrict the run to just those tiles")


def restrict_keys(keys, keys_file):
    """Keep only keys whose tile id is listed in keys_file (None = keep all)."""

    if keys_file is None:
        return keys
    if keys_file.startswith("s3://"):
        # lets a Batch container read the list (a local path isn't visible there)
        bucket, key = keys_file.removeprefix("s3://").split("/", 1)
        text = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        lines = text.splitlines()
    else:
        with open(keys_file) as f:
            lines = f.read().splitlines()
    wanted = {tile_id_from_key(line.strip()) for line in lines if line.strip()}
    kept = [k for k in keys if tile_id_from_key(k) in wanted]
    print(f"--keys-file: {len(kept):,} of {len(keys):,} tiles selected ({len(wanted):,} listed)")
    return kept
