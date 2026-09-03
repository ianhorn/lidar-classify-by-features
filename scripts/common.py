#!/usr/bin/env python3

"""
Shared constants/helpers for the building-classification scripts.
FEATURE_COLUMNS must stay identical between training and inference -- not
because order matters (XGBoost's sklearn API matches by column name, not
position), but because the *set* must match exactly, or predict() raises
on an unseen/missing column.
"""

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
