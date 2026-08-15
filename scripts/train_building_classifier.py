#!/usr/bin/env python3

"""
First-pass building-point classifier, trained on a sample of the features
parquet files already sitting in s3://lidar-classification/phase2/features/.

This is meant to be read end to end, not just run -- each step is kept
separate and printed so you can see what's happening at every stage. Once
you're comfortable with the shape of it, this is the piece to start
rewriting yourself.

Label: WithinFootprint (from the Overture building footprint join already
done in the pipeline) is used directly as "is this a building point,"
without any refinement. It's a noisy label -- Overture's footprints aren't
captured at the same time as the lidar, so some True/False here will be
wrong wherever a building was built or demolished between the two. Good
enough to get a first model and see how it performs; refining the label
rule (e.g. requiring some minimum HAG too) is a natural next step once you
can see where this one is actually wrong.

Features: only the geometry/reflectance columns go in as model input --
not X/Y/Z (a model trained on absolute coordinates overfits to specific
tile locations instead of learning geometry that generalizes), and not
WithinFootprint/FootprintHeight (that's the label itself, not a legitimate
input -- including it would leak the answer directly into the features).
"""

import random

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

S3_BUCKET = "lidar-classification"
FEATURES_PREFIX = "phase2/features/"

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


def load_sample(keys, n_tiles=25, row_frac=0.05, random_state=0):
    """
    Read n_tiles random tiles' features, keeping row_frac of each tile's
    rows. Full tiles are millions of rows each -- for a first pass this
    keeps the whole thing fast enough to iterate on in minutes, not hours.
    Each row is tagged with source_tile so a later train/test split can
    hold out entire tiles, not just random rows (points within one tile
    are spatially correlated, so a random row split would leak).
    """

    rng = random.Random(random_state)
    sample_keys = rng.sample(keys, min(n_tiles, len(keys)))

    frames = []
    for key in sample_keys:
        tile_id = key.rsplit("/", 1)[-1].removesuffix(".parquet")
        tile_df = pd.read_parquet(f"s3://{S3_BUCKET}/{key}")
        tile_df = tile_df.sample(frac=row_frac, random_state=random_state)
        tile_df["source_tile"] = tile_id
        frames.append(tile_df)
        print(f"loaded {len(tile_df):,} rows from {tile_id}")

    return pd.concat(frames, ignore_index=True)


def main():
    print("listing available feature tiles in S3...")
    keys = list_feature_tiles()
    print(f"{len(keys):,} feature tiles available")

    df = load_sample(keys)
    print(f"\n{len(df):,} total rows loaded from {df['source_tile'].nunique()} tiles")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].astype(int)
    groups = df["source_tile"]

    # Held-out tiles, not held-out rows -- tests whether the model
    # generalizes to *unseen areas*, which is the real question, not
    # whether it memorized these specific 25 tiles.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"train: {len(X_train):,} rows from {groups.iloc[train_idx].nunique()} tiles")
    print(f"test:  {len(X_test):,} rows from {groups.iloc[test_idx].nunique()} tiles")

    print("\ntraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        # Not -1 (all cores) -- matches this machine's other branches, which
        # reserve some cores so it stays usable for other things, and this
        # may run alongside the main pipeline rather than instead of it.
        n_jobs=20,
        random_state=0,
    )
    model.fit(X_train, y_train)

    print("\n--- test set performance ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["not building", "building"]))

    print("--- feature importances ---")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).to_string())

    model_path = "scripts/building_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"\nsaved model to {model_path}")


if __name__ == "__main__":
    main()
