#!/usr/bin/env python3

############################################################
# Work out what changed upstream (KyFromAbove Phase2) since our
# outputs were built, and write a manifest of tiles to (re)process
# plus a list of orphaned outputs to delete. Read-only against AWS;
# writes only to ./stac_stage/ (see apply_stac_changes.py).
#
# The bucket listing is the source of truth, NOT the STAC catalog:
# as of 2026-11 the STAC still lists 93 *.copz ids whose files were
# replaced by *.copc on 2026-08-31 (so their hrefs 404), and lacks
# 927 tiles that exist in the bucket (see phase2_missing_tile_keys.txt).
# STAC is used only to reuse bbox/href where it is accurate; for
# everything else the bbox comes from the COPC header.
#
# Every bucket tile gets one action, keyed on its normalized id
# (copz -> copc, same normalization every S3 output key uses):
#   new      no features parquet, not flagged problematic/unprocessable
#   replaced features parquet exists but the source .laz is newer
#   retry    flagged problematic/unprocessable, but the source is newer
#            than the flag (the replacement probably fixed it)
#   ok       nothing to do
# and every output id with NO bucket object is "orphaned" (its laz/
# buildings/features/classified outputs should be deleted).
############################################################

import argparse
import io
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd
import requests
from pyproj import Transformer

SRC_BUCKET = "kyfromabove"
SRC_PREFIX = "elevation/PointCloud/Phase2/"
OUT_BUCKET = "lidar-classification"
STAC_URL = "https://drwgni8q1h.execute-api.us-west-2.amazonaws.com"
COLLECTION = "laz-phase2"
SRC_HTTP = f"https://{SRC_BUCKET}.s3.us-west-2.amazonaws.com/{SRC_PREFIX}"
SRC_CRS = 3089  # NAD83 / Kentucky Single Zone (ftUS) -- per the STAC items' proj:wkt2


def normalize(item_id):
    return item_id.replace("copz", "copc")


def list_objects(bucket, prefix):
    s3 = boto3.client("s3")
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out.append((o["Key"], o["LastModified"], o["Size"]))
    return pd.DataFrame(out, columns=["key", "last_modified", "size"])


def fetch_stac_bboxes(base_url):
    url = f"{base_url}/collections/{COLLECTION}/items?limit=1000"
    rows, session = [], requests.Session()
    while url:
        r = session.get(url, timeout=120)
        r.raise_for_status()
        body = r.json()
        rows += [{"stac_id": f["id"], "bbox": f["bbox"]} for f in body["features"]]
        url = next((l["href"] for l in body.get("links", []) if l["rel"] == "next"), None)
    print(f"{len(rows):,} STAC items")
    return {normalize(r["stac_id"]): r["bbox"] for r in rows}


def copc_header_extent(filename):
    """Read min/max X/Y (native CRS) from the LAS public header via one range request."""
    r = requests.get(SRC_HTTP + filename, headers={"Range": "bytes=0-374"}, timeout=60)
    r.raise_for_status()
    max_x, min_x, max_y, min_y = struct.unpack_from("<4d", r.content, 179)
    return min_x, min_y, max_x, max_y


def to_4326(extent):
    # Transformer use stays on the main thread: pyproj transformers are not
    # thread-safe, and doing this inside the header-fetch pool silently
    # produced inf for ~60% of tiles.
    t = Transformer.from_crs(SRC_CRS, 4326, always_xy=True)
    min_x, min_y, max_x, max_y = extent
    (x0, y0), (x1, y1) = t.transform(min_x, min_y), t.transform(max_x, max_y)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def copc_bbox_4326(filename):
    return to_4326(copc_header_extent(filename))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="stac_stage")
    parser.add_argument("--header-workers", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.out_dir
    out_dir.mkdir(exist_ok=True)

    # --- bucket = truth -----------------------------------------------------
    src = list_objects(SRC_BUCKET, SRC_PREFIX)
    src = src[src["key"].str.endswith(".laz")]
    src["filename"] = src["key"].str.removeprefix(SRC_PREFIX)
    src = src[~src["filename"].str.contains("/")].copy()  # skips laztoprocess/ etc.
    src["id"] = src["filename"].str.removesuffix(".laz")
    src["output_id"] = src["id"].map(normalize)
    dupes = src[src.duplicated("output_id", keep=False)]
    if len(dupes):
        # copz and copc twins both present: keep the newer upload
        print(f"WARNING: {len(dupes)} bucket objects collide after copz->copc; keeping newest:\n{dupes[['filename','last_modified']]}")
        src = src.sort_values("last_modified").drop_duplicates("output_id", keep="last")
    print(f"{len(src):,} tiles in bucket")

    # --- what we've produced ----------------------------------------------------
    feats = list_objects(OUT_BUCKET, "phase2/features/")
    feats["output_id"] = feats["key"].str.rsplit("/", n=1).str[-1].str.removesuffix(".parquet")
    prob = list_objects(OUT_BUCKET, "phase2/laz-problematic/")
    prob["output_id"] = prob["key"].str.rsplit("/", n=1).str[-1].str.removesuffix("_problematic.parquet")
    prob = prob[prob["key"].str.endswith("_problematic.parquet")]
    unproc = list_objects(OUT_BUCKET, "phase2/laz-unprocessable/")
    unproc["output_id"] = unproc["key"].str.rsplit("/", n=1).str[-1].str.removesuffix("_unprocessable.parquet")
    unproc = unproc[unproc["key"].str.endswith("_unprocessable.parquet")]
    flags = pd.concat([prob, unproc]).groupby("output_id")["last_modified"].max().rename("flag_modified")

    m = src.merge(feats[["output_id", "last_modified"]].rename(columns={"last_modified": "features_modified"}),
                  on="output_id", how="left")
    m = m.merge(flags, on="output_id", how="left")

    # A problematic flag records the bad href's filename. If that file is no
    # longer in the bucket, the flag is stale (e.g. the old *.copz was
    # replaced by *.copc on 2026-08-31, so the href 404'd and was flagged
    # even though the tile itself is fine now).
    bucket_files = set(src["filename"])
    gone_flagged = set()
    def read_flag_filenames(key):
        body = boto3.client("s3").get_object(Bucket=OUT_BUCKET, Key=key)["Body"]
        return pd.read_parquet(io.BytesIO(body.read()))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for oid, df in zip(prob["output_id"], pool.map(read_flag_filenames, prob["key"])):
            if not set(df["filename"]) <= bucket_files:
                gone_flagged.add(oid)

    def action(r):
        if pd.notna(r.features_modified):
            return "replaced" if r.last_modified > r.features_modified else "ok"
        if r.output_id in gone_flagged or (pd.notna(r.flag_modified) and r.last_modified > r.flag_modified):
            return "retry"
        return "new" if pd.isna(r.flag_modified) else "ok"
    m["action"] = m.apply(action, axis=1)
    # A tile flagged problematic/unprocessable with no newer source stays "ok"
    # (nothing to do); count it separately for the summary.
    m["flagged"] = m["flag_modified"].notna()

    # --- orphaned outputs (ours, but the bucket no longer has the tile) ----------
    orphaned = sorted((set(feats["output_id"]) | set(flags.index)) - set(src["output_id"]))

    # --- bbox: STAC where it has the tile, else read the COPC header -------------
    stac = fetch_stac_bboxes(STAC_URL)
    bboxes = {oid: stac[oid] for oid in m["output_id"] if oid in stac}
    bad_header = []
    need = m[~m["output_id"].isin(stac)]
    print(f"{len(need):,} bucket tiles are missing from STAC; reading their COPC headers for bbox")
    with ThreadPoolExecutor(max_workers=args.header_workers) as pool:
        for oid, extent in zip(need["output_id"], pool.map(copc_header_extent, need["filename"])):
            if extent == (0.0, 0.0, 0.0, 0.0):
                bad_header.append(oid)  # all-zero LAS header: no bounds, can't be tiled/cropped
            else:
                bboxes[oid] = to_4326(extent)
    # Kentucky is roughly x -90..-81.9, y 36.4..39.2; a tile is ~0.015 deg wide.
    for oid, (x0, y0, x1, y1) in bboxes.items():
        assert -90.7 < x0 < x1 < -81.5 and 36.0 < y0 < y1 < 39.5 and (x1 - x0) < 0.1 and (y1 - y0) < 0.1, \
            f"implausible bbox for {oid}: {(x0, y0, x1, y1)}"
    (out_dir / "bad_header.txt").write_text("\n".join(sorted(bad_header)) + "\n")
    print(f"{len(bad_header)} tiles have an all-zero header -> excluded (stac_stage/bad_header.txt)")
    m["bbox"] = m["output_id"].map(bboxes)
    m = m[m["bbox"].notna()].copy()
    todo = m[m["action"] != "ok"].copy()
    print(f"{len(todo):,} tiles to process")
    # sanity: header-derived bbox must agree with STAC's for tiles that have both
    check = m[(m["action"] == "ok") & m["output_id"].isin(stac)].head(5)
    for fn, oid in zip(check["filename"], check["output_id"]):
        h, s = copc_bbox_4326(fn), stac[oid]
        assert all(abs(a - b) < 1e-3 for a, b in zip(h, s)), f"header bbox mismatch for {fn}: {h} vs {s}"
    print("header bbox matches STAC bbox on 5 spot-checked tiles")

    todo["href"] = SRC_HTTP + todo["filename"]
    for i, c in enumerate(["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]):
        todo[c] = todo["bbox"].str[i]
    manifest = todo[["output_id", "id", "href", "action", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]]
    manifest = manifest.rename(columns={"id": "source_id"}).reset_index(drop=True)
    manifest.to_parquet(out_dir / "manifest.parquet", compression="zstd")
    (out_dir / "orphaned.txt").write_text("\n".join(orphaned) + ("\n" if orphaned else ""))

    # all bucket tiles, for neighbor lookups when buffering (includes "ok" tiles)
    allt = m[["output_id", "filename"]].copy()
    allt["href"] = SRC_HTTP + allt["filename"]
    allt["bbox"] = m["bbox"]
    allt.to_parquet(out_dir / "bucket_tiles.parquet")

    print("\naction counts:\n" + m["action"].value_counts().to_string())
    print(f"already-flagged and unchanged (nothing to do): {((m['action'] == 'ok') & m['flagged']).sum()}")
    print(f"orphaned outputs (no bucket tile): {len(orphaned)}")
    print(f"wrote {out_dir}/manifest.parquet ({len(manifest)} rows), orphaned.txt")


if __name__ == "__main__":
    main()
