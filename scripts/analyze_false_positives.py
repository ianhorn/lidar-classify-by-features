#!/usr/bin/env python3

"""
Diagnose where the building classifier's disagreements with the Overture
footprint label come from, so feature work targets the biggest error class
instead of a guess. Read-only: pulls a sample of feature tiles from S3, runs
the model, and prints/saves summary tables. Nothing is written to S3.

A "false positive" here means the model says building (probability >= 0.5 and
HAG > floor, same rule as classify_tiles.py) but WithinFootprint is False.
Some of those are label errors (the footprint is older/newer than the lidar),
not model errors -- so read the tables as "what do the disagreements look
like", not "what is wrong with the model".

Tables:
  - FP / FN share by HAG band, NumberOfReturns, CanopyHeight, source
    Classification (ground=2, veg=3-5, building=6, noise=7/18, ...).
  - FP cluster size: false-positive points are rasterised to a 1 m grid and
    connected blobs are measured. Vehicles are small blobs (a few dozen ft^2); canopy
    is large and ragged; a mislabeled real building is a large compact blob.
    That split tells us whether a blob-size feature would help.
"""

import argparse
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import ndimage

from common import FEATURE_COLUMNS, LABEL_COLUMN, S3_BUCKET, list_feature_tiles, tile_id_from_key

COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN, "X", "Y", "Z", "Classification"]
HAG_BINS = [-np.inf, 2, 5, 10, 20, 40, np.inf]
CLUSTER_BINS = [0, 50, 200, 500, 2000, 10000, np.inf]  # ft^2


def analyze_tile(key, model, hag_floor, cell):
    df = pd.read_parquet(f"s3://{S3_BUCKET}/{key}", columns=COLUMNS)
    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["pred"] = (proba >= 0.5) & (df["HAG"] > hag_floor)
    label = df[LABEL_COLUMN].astype(bool)
    df["fp"], df["fn"] = df["pred"] & ~label, ~df["pred"] & label

    # Connected blobs of *predicted building* on a cell-size grid (cell is in
    # CRS units, ftUS; areas stay in ft^2, matching the data). Per blob: area (ft^2), how many points agree with the
    # footprint label (tp) and how many don't (fp). That shows whether a
    # "drop blobs smaller than X" rule would remove mostly errors or mostly
    # real building points.
    pr = df[df["pred"]]
    blobs = pd.DataFrame(columns=["area", "tp", "fp"])
    if len(pr):
        x0, y0 = df["X"].min(), df["Y"].min()
        ix = ((pr["X"] - x0) / cell).astype(int).to_numpy()
        iy = ((pr["Y"] - y0) / cell).astype(int).to_numpy()
        grid = np.zeros((ix.max() + 1, iy.max() + 1), dtype=bool)
        grid[ix, iy] = True
        labeled, n = ndimage.label(grid)
        ids = labeled[ix, iy]
        isfp = pr["fp"].to_numpy()
        cells = np.bincount(labeled[grid], minlength=n + 1)
        blobs = pd.DataFrame({
            "area": cells[1:] * cell ** 2,
            "tp": np.bincount(ids[~isfp], minlength=n + 1)[1:],
            "fp": np.bincount(ids[isfp], minlength=n + 1)[1:],
        })
    sizes = blobs

    keep = ["HAG", "NumberOfReturns", "CanopyHeight", "Classification", "fp", "fn", "pred", LABEL_COLUMN]
    out = df[keep].copy()
    out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(bool)
    # downsample the huge non-error majority so the combined frame fits in RAM
    errors = out[out["fp"] | out["fn"]]
    rest = out[~(out["fp"] | out["fn"])].sample(frac=0.02, random_state=0)
    rest["w"], errors = 50.0, errors.assign(w=1.0)
    return pd.concat([errors, rest]), sizes


def share_table(df, by, name):
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({
        "points": g["w"].sum(),
        "fp": g.apply(lambda x: (x["fp"] * x["w"]).sum(), include_groups=False),
        "fn": g.apply(lambda x: (x["fn"] * x["w"]).sum(), include_groups=False),
    })
    t["fp_of_all_fp_%"] = 100 * t["fp"] / t["fp"].sum()
    t["fn_of_all_fn_%"] = 100 * t["fn"] / t["fn"].sum()
    t["fp_rate_%"] = 100 * t["fp"] / t["points"]
    print(f"\n=== by {name} ===\n{t.round(2).to_string()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="building_classifier_v4.json")
    parser.add_argument("--n-tiles", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--hag-floor", type=float, default=2.0)
    parser.add_argument("--cell", type=float, default=3.0, help="blob grid cell edge in ftUS (the tile CRS units)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    model.set_params(device=args.device)

    keys = random.Random(args.random_state).sample(list_feature_tiles(), args.n_tiles)
    frames, all_sizes = [], []
    for key in keys:
        df, sizes = analyze_tile(key, model, args.hag_floor, args.cell)
        frames.append(df)
        all_sizes.append(sizes)
        print(f"{tile_id_from_key(key)}: {int(df.loc[df['fp'], 'w'].sum()):,} FP, "
              f"{int(df.loc[df['fn'], 'w'].sum()):,} FN, {len(sizes)} predicted blobs")

    d = pd.concat(frames, ignore_index=True)
    d["HAG band (ft)"] = pd.cut(d["HAG"], HAG_BINS)
    d["NumberOfReturns"] = d["NumberOfReturns"].clip(upper=4)
    d["canopy"] = pd.cut(d["CanopyHeight"], [-np.inf, 0, 3, 10, np.inf]).astype(str).replace("nan", "no data")
    print(f"\nall points: {d['w'].sum():,.0f}  FP: {(d['fp'] * d['w']).sum():,.0f}  FN: {(d['fn'] * d['w']).sum():,.0f}")
    for col in ["HAG band (ft)", "NumberOfReturns", "canopy", "Classification"]:
        share_table(d, col, col)

    blobs = pd.concat(all_sizes, ignore_index=True)
    blobs["bin"] = pd.cut(blobs["area"], CLUSTER_BINS)
    t = blobs.groupby("bin", observed=True).agg(blobs=("area", "size"), tp_pts=("tp", "sum"), fp_pts=("fp", "sum"))
    t["fp_share_%"] = 100 * t["fp_pts"] / (t["tp_pts"] + t["fp_pts"])
    print(f"\n=== predicted-building blobs by area (ft^2) ===\n{t.round(1).to_string()}")
    print("\n=== effect of dropping blobs smaller than X ft^2 ===")
    for x in [25, 50, 100, 200, 500, 1000]:
        small = blobs[blobs["area"] < x]
        print(f"< {x:>4} ft^2: removes {100 * small['fp'].sum() / blobs['fp'].sum():5.1f}% of FP, "
              f"{100 * small['tp'].sum() / blobs['tp'].sum():5.1f}% of TP")

if __name__ == "__main__":
    main()
