#!/usr/bin/env python3

"""
CanopyHeight feature from Meta/WRI's canopy height raster (clipped to
Kentucky), sampled directly out of the vendor's FileGDB raster -- see
canopy-files/raw/ -- with no format conversion. Sampled per-point from each
tile's X/Y; the raster stays in its native CRS (EPSG:3857) and points are
reprojected to match it, not the other way around -- warping the whole
statewide raster to the lidar CRS (EPSG:3089) once would mean transforming
~320 billion pixels up front when any one tile only ever queries a tiny
fraction of them. Reprojecting a tile's points instead costs one cheap
vectorized pyproj transform per tile, negligible next to the I/O these
backfill scripts are already bound by.

Uses osgeo.gdal directly rather than rasterio: rasterio ships its own
bundled libgdal (via its wheel), which doesn't include the OpenFileGDB
raster driver this geodatabase needs, so it can't open the .gdb at all.
osgeo.gdal here is the conda env's own GDAL build (same one gdalinfo/
gdal_translate use), which does have that driver.

Run directly on one features parquet to prototype/inspect a tile:

    python canopy_features.py /path/to/tile.parquet --out-dir /mnt/d/Data/bridge_test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import scipy.ndimage
from osgeo import gdal

gdal.UseExceptions()

RASTER_PATH = "/home/ian/repos/lidar-classify-by-features/canopy-files/raw/Ky_Meta_CanopyHeightModel.gdb"
LIDAR_CRS = "EPSG:3089"
RASTER_CRS = "EPSG:3857"

# +/-1 pixel (a 3x3 window, ~3.6m across at this raster's ~1.19m pixel size).
# Sampling only the single nearest pixel turned out to be too exact a match
# to ask of a raster from a different source/date than the lidar capture --
# a 30-tile QA pass found 88% of the trained model's building false
# positives sat at CanopyHeight==0 despite tree-like geometry (low
# Planarity, high SurfaceVariation/Linearity, multiple returns), almost all
# of them within a pixel or two of a neighboring canopy pixel. Taking the
# max over a small window catches "near canopy" even when the exact point
# falls in a registration-gap pixel, without washing out genuinely open
# ground far from any canopy the way a larger window or a mean would.
WINDOW_RADIUS = 1

_to_raster_crs = pyproj.Transformer.from_crs(LIDAR_CRS, RASTER_CRS, always_xy=True)


def compute_canopy_height(x, y, raster_path=RASTER_PATH, window_radius=WINDOW_RADIUS):
    """
    CanopyHeight (meters) at each lidar point's X/Y: the max raster value in
    a (2*window_radius+1)-pixel-square window around the point, via a
    windowed read of just the raster area the tile's points cover -- not the
    whole statewide raster. NaN where no pixel in that window has valid data
    (XGBoost handles NaN as a native missing-value marker, so this needs no
    sentinel).

    Opens its own dataset handle per call rather than sharing one across
    threads -- GDAL dataset objects aren't safe to read concurrently from
    multiple threads, and this is called from a ThreadPoolExecutor worker
    per tile (see backfill_canopy_features.py).
    """

    rx, ry = _to_raster_crs.transform(x, y)

    ds = gdal.Open(raster_path)
    origin_x, pixel_w, _, origin_y, _, pixel_h = ds.GetGeoTransform()
    raster_w, raster_h = ds.RasterXSize, ds.RasterYSize

    # Axis-aligned raster (no rotation terms), so pixel/line from world
    # coordinates is a plain affine scale+offset -- no matrix inverse needed.
    cols = (rx - origin_x) / pixel_w
    rows = (ry - origin_y) / pixel_h

    # Pad the window read by window_radius (for the neighborhood max below)
    # plus 1 more pixel (points right at the padded edge would otherwise
    # round/filter outside the block read here) on each side, clipped to the
    # raster's own extent.
    pad = window_radius + 1
    col_lo = max(int(np.floor(cols.min())) - pad, 0)
    col_hi = min(int(np.ceil(cols.max())) + pad, raster_w - 1)
    row_lo = max(int(np.floor(rows.min())) - pad, 0)
    row_hi = min(int(np.ceil(rows.max())) + pad, raster_h - 1)
    win_w, win_h = col_hi - col_lo + 1, row_hi - row_lo + 1

    band = ds.GetRasterBand(1)
    block = band.ReadAsArray(col_lo, row_lo, win_w, win_h)
    mask_block = band.GetMaskBand().ReadAsArray(col_lo, row_lo, win_w, win_h) != 0
    ds = None  # closes the handle

    # -inf for invalid pixels so they never win the max filter; a separate
    # filter over the validity mask tracks whether *any* pixel in a given
    # window was valid, so "no valid data nearby" still comes out as NaN
    # rather than -inf.
    size = 2 * window_radius + 1
    masked = np.where(mask_block, block.astype(np.float32), -np.inf)
    window_max = scipy.ndimage.maximum_filter(masked, size=size, mode="nearest")
    window_valid = scipy.ndimage.maximum_filter(mask_block.astype(np.uint8), size=size, mode="nearest") != 0

    local_cols = np.clip(np.round(cols).astype(np.int64) - col_lo, 0, win_w - 1)
    local_rows = np.clip(np.round(rows).astype(np.int64) - row_lo, 0, win_h - 1)

    height = window_max[local_rows, local_cols]
    valid = window_valid[local_rows, local_cols]
    height = np.where(valid, height, np.nan).astype(np.float32)
    return height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", help="a features parquet with X/Y columns")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet, columns=["X", "Y", "Z", "HAG", "Classification", "WithinFootprint"])
    x, y = df["X"].to_numpy(), df["Y"].to_numpy()
    print(f"{len(df):,} points")

    height = compute_canopy_height(x, y)
    df["CanopyHeight"] = height

    valid = ~np.isnan(height)
    print(f"valid canopy data: {valid.sum():,} points ({valid.mean():.2%})")
    print(f"canopy height (meters) percentiles where valid:")
    print(df.loc[valid, "CanopyHeight"].describe(percentiles=[0.1, 0.5, 0.9]).round(2))

    in_canopy = valid & (height > 2)
    print(f"\nHAG percentiles, in canopy (>2m) vs. not:")
    print(pd.DataFrame({
        "in_canopy": df.loc[in_canopy, "HAG"].describe(percentiles=[0.1, 0.5, 0.9]),
        "elsewhere": df.loc[~in_canopy, "HAG"].describe(percentiles=[0.1, 0.5, 0.9]),
    }).round(1))

    stem = Path(args.parquet).name.split("_")[0]
    out_path = out_dir / f"{stem}_with_canopy.parquet"
    df.to_parquet(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
