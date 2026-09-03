#!/usr/bin/env python3

"""
First-pass building-point classifier, trained on a stratified sample of
the features parquet files already sitting in
s3://lidar-classification/phase2/features/ (38,000+ tiles at time of
writing -- far too much to pull down whole, hence the sampling below).

Label: WithinFootprint (from the Overture building footprint join already
done in the main pipeline) is used directly as "is this a building point,"
without any refinement. It's a noisy label -- Overture's footprints aren't
captured at the same time as the lidar, so some True/False here will be
wrong wherever a building was built or demolished between the two capture
dates. Good enough for a first model.

Sampling: WithinFootprint's share of points varies a lot by tile -- under
1% on a mostly-rural tile, but a dense urban/commercial tile can be
90%+ building. A flat random row sample would starve rural tiles of
building examples, so each sampled tile instead keeps every
building-labeled row and caps the *non-building* rows at
MAX_NONBUILDING_PER_TILE. Note this means the reverse can also happen on a
building-heavy tile -- it's the minority class (non-building) that gets
capped there, not building -- so scale_pos_weight below is computed from
the actual resulting train-split ratio, not assumed in advance.

Every point a tile has -- including ones the LAS source already tagged
Classification 7/18 (noise) -- goes into the eligible pool here. That
existing noise tagging is suspected to be wrong in places, so nothing in
this pipeline filters on it; the model judges every point on its own
geometry instead of deferring to a flag that might be incorrect.

Features: only the geometry/reflectance columns (scripts/common.py's
FEATURE_COLUMNS) go in as model input -- not X/Y/Z (a model trained on
absolute coordinates overfits to specific tile locations instead of
learning geometry that generalizes), and not WithinFootprint itself
(that's the label -- including it would leak the answer directly into the
features).

Memory note: fetch_tile_sample has to read the full feature+label columns
for a tile before it can sample it (needs every row's label to know which
rows are building rows). A dense tile's worth of those columns can run
multiple GB in flight, so --fetch-workers concurrency should be sized to
available RAM, not just I/O parallelism -- keep it low on a machine
sharing memory with other things (e.g. a WSL VM with a capped memory
allocation) and raise it only after confirming headroom.
"""

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

from common import FEATURE_COLUMNS, LABEL_COLUMN, S3_BUCKET, list_feature_tiles, tile_id_from_key

DEFAULT_N_TILES = 100
DEFAULT_MAX_NONBUILDING_PER_TILE = 40_000
DEFAULT_FETCH_WORKERS = 3  # I/O bound, but each in-flight tile can run multiple GB -- see module docstring
DEFAULT_MODEL_PATH = "scripts/building_classifier.json"


def fetch_tile_sample(key, max_nonbuilding, random_state):
    """
    Read one tile's feature/label columns and stratify-sample it: every
    building row, plus a random sample of non-building rows capped at
    max_nonbuilding. Reading the feature/label columns for the whole tile
    is unavoidable (need every row's label to know which are building
    rows before sampling), so this is dominated by S3 transfer time, not
    local compute.
    """

    path = f"s3://{S3_BUCKET}/{key}"
    df = pd.read_parquet(path, columns=FEATURE_COLUMNS + [LABEL_COLUMN])

    building = df[df[LABEL_COLUMN]]
    nonbuilding = df[~df[LABEL_COLUMN]]
    if len(nonbuilding) > max_nonbuilding:
        nonbuilding = nonbuilding.sample(n=max_nonbuilding, random_state=random_state)

    tile_id = tile_id_from_key(key)
    sample = pd.concat([building, nonbuilding], ignore_index=True)
    sample["source_tile"] = tile_id
    return tile_id, sample


def load_sample(keys, n_tiles, max_nonbuilding, fetch_workers, random_state):
    """Stratify-sample n_tiles random tiles, fetched concurrently to overlap S3 wait time."""

    rng = random.Random(random_state)
    sample_keys = rng.sample(keys, min(n_tiles, len(keys)))

    frames = []
    with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
        futures = {
            executor.submit(fetch_tile_sample, key, max_nonbuilding, random_state): key
            for key in sample_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                tile_id, tile_df = future.result()
            except Exception as e:
                print(f"ERROR fetching {key}: {e}")
                continue
            frames.append(tile_df)
            n_building = int(tile_df[LABEL_COLUMN].sum())
            print(f"loaded {len(tile_df):,} rows ({n_building:,} building) from {tile_id}")

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tiles", type=int, default=DEFAULT_N_TILES, help="number of tiles to sample")
    parser.add_argument("--max-nonbuilding", type=int, default=DEFAULT_MAX_NONBUILDING_PER_TILE,
                         help="cap on non-building rows kept per tile")
    parser.add_argument("--fetch-workers", type=int, default=DEFAULT_FETCH_WORKERS)
    parser.add_argument("--test-size", type=float, default=0.25, help="fraction of tiles (not rows) held out")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    print("listing available feature tiles in S3...")
    keys = list_feature_tiles()
    print(f"{len(keys):,} feature tiles available")

    start = time.perf_counter()
    df = load_sample(keys, args.n_tiles, args.max_nonbuilding, args.fetch_workers, args.random_state)
    elapsed = time.perf_counter() - start
    print(f"\n{len(df):,} total rows loaded from {df['source_tile'].nunique()} tiles in {elapsed:.1f}s")
    print(f"building fraction in sample: {df[LABEL_COLUMN].mean():.3f}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].astype(int)
    groups = df["source_tile"]

    # Held-out tiles, not held-out rows -- tests whether the model
    # generalizes to *unseen areas*, which is the real question, not
    # whether it memorized these specific tiles.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"train: {len(X_train):,} rows from {groups.iloc[train_idx].nunique()} tiles")
    print(f"test:  {len(X_test):,} rows from {groups.iloc[test_idx].nunique()} tiles")

    print(f"\ntraining XGBClassifier on {args.device}...")
    # scale_pos_weight rebalances the loss toward the positive class on
    # top of the row-level stratification in fetch_tile_sample -- the
    # sampled set is still whatever ratio the capping above happened to
    # produce, not necessarily balanced, so this reads the actual training
    # split's ratio rather than assuming one.
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        device=args.device,
        tree_method="hist",
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    print("\n--- test set performance ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["not building", "building"]))

    print("--- feature importances ---")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).to_string())

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"\nsaved model to {model_path}")


if __name__ == "__main__":
    main()
