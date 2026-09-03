#!/usr/bin/env python3

"""
Apply a trained building-point classifier (see
scripts/train_building_classifier.py) to feature tiles in
s3://lidar-classification/phase2/features/, writing each tile's
predictions back to s3://lidar-classification/phase2/classified/.

Every point in a tile is classified -- including points the LAS source
already tagged Classification 7/18 (noise). That tagging is suspected to
be wrong in places, so nothing here treats it as a reason to skip a
point; the model judges every point from its own geometry.

Resume support mirrors src/main.py: a tile whose classified output
already exists on S3 is skipped unless --overwrite is passed, so a
re-run over the same tile list doesn't redo completed work.

This reads every row of every tile (no sampling, unlike training), so at
the throughput measured locally (~100 tiles/108min with 3 concurrent
workers) a full 38,000+ tile run would take on the order of 1-2 weeks on
a single machine. classify_tile() below is written to be reusable as-is
inside a Coiled/Dask submit() call the same way src/main.py's process()
is, for exactly that reason.
"""

import argparse
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import xgboost as xgb

from common import FEATURE_COLUMNS, S3_BUCKET, list_feature_tiles, s3_object_exists, tile_id_from_key, upload_to_s3

CLASSIFIED_PREFIX = "phase2/classified/"

# Kept in the output for context/QA -- not fed to the model.
CONTEXT_COLUMNS = ["X", "Y", "Z", "Classification", "WithinFootprint"]

DEFAULT_MODEL_PATH = "scripts/building_classifier.json"
DEFAULT_WORKERS = 3  # I/O bound, but each in-flight tile can run multiple GB -- see train_building_classifier.py


def classify_tile(key, model, overwrite):
    """
    Read one tile's features from S3, score every point, and upload a
    parquet with the context columns plus BuildingProbability/IsBuildingML
    back to S3. model.predict_proba is called from multiple threads here
    (see main()) -- XGBoost's own docs guarantee concurrent *reads*
    (predict) on one Booster are safe; only concurrent *mutation* (fit)
    isn't, and nothing here calls fit.
    """

    tile_id = tile_id_from_key(key)
    out_key = f"{CLASSIFIED_PREFIX}{tile_id}.parquet"

    if not overwrite and s3_object_exists(S3_BUCKET, out_key):
        return tile_id, "skipped", None

    path = f"s3://{S3_BUCKET}/{key}"
    df = pd.read_parquet(path, columns=FEATURE_COLUMNS + CONTEXT_COLUMNS)

    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["BuildingProbability"] = proba
    df["IsBuildingML"] = proba >= 0.5

    # Quick eyeball stats, computed here while the data's already in
    # memory rather than a separate re-read of the uploaded output.
    # WithinFootprint agreement is a sanity check, not ground truth -- it's
    # the same noisy label the model was trained on, so disagreement isn't
    # automatically the model being wrong.
    stats = {
        "building_fraction": float(df["IsBuildingML"].mean()),
        "mean_probability": float(df["BuildingProbability"].mean()),
        "footprint_agreement": float((df["IsBuildingML"] == df["WithinFootprint"]).mean()),
    }

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path)
    upload_to_s3(tmp_path, S3_BUCKET, out_key)
    tmp_path.unlink()

    return tile_id, "ok", stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-tiles", type=int, default=None,
                         help="classify a random sample of N tiles instead of every available tile (for testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true", help="reclassify tiles that already have output on S3")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    model.set_params(device=args.device)

    print("listing feature tiles in S3...")
    keys = list_feature_tiles()
    print(f"{len(keys):,} feature tiles available")

    if args.n_tiles is not None:
        keys = random.Random(args.random_state).sample(keys, min(args.n_tiles, len(keys)))
        print(f"sampled {len(keys)} tiles for this run")

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(classify_tile, key, model, args.overwrite): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            tile_id = tile_id_from_key(key)
            try:
                tile_id, status, stats = future.result()
            except Exception as e:
                print(f"ERROR classifying {key}: {e}")
                results.append((tile_id, "error", "", "", "", str(e)))
                continue
            if stats is None:
                print(f"{tile_id}: {status}")
                results.append((tile_id, status, "", "", "", ""))
            else:
                print(
                    f"{tile_id}: {status}  building_fraction={stats['building_fraction']:.4f}  "
                    f"mean_probability={stats['mean_probability']:.4f}  "
                    f"footprint_agreement={stats['footprint_agreement']:.4f}"
                )
                results.append((
                    tile_id, status,
                    stats["building_fraction"], stats["mean_probability"], stats["footprint_agreement"],
                    "",
                ))

    elapsed = time.perf_counter() - start
    print(f"\nprocessed {len(results)} tiles in {elapsed:.1f}s")

    results_df = pd.DataFrame(
        results,
        columns=["tile_id", "status", "building_fraction", "mean_probability", "footprint_agreement", "error"],
    )
    local_path = Path(tempfile.gettempdir()) / "classify_results.parquet"
    results_df.to_parquet(local_path)
    upload_to_s3(local_path, S3_BUCKET, "phase2/classify_results.parquet")
    local_path.unlink()


if __name__ == "__main__":
    main()
