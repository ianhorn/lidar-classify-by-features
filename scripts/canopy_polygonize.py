#!/usr/bin/env python3

"""
Polygonize Meta/WRI's canopy height raster (see canopy_features.py) into
canopy-present polygons (no distinction by height, just canopy vs. not),
clipped to Kentucky's state boundary plus a 1000ft buffer -- for publishing
as an AGOL layer.

Ground/no-canopy pixels (height == 0) are excluded entirely, as are pixels
outside the buffered boundary or outside the raster's own valid-data mask --
all three conditions are folded into one mask before polygonizing, so no
"no canopy" polygons ever get created in the first place (nothing to filter
out afterward). Polygonizing a single binary canopy/not-canopy mask, rather
than the raw per-meter integer heights, matters a lot here: natural canopy
height varies almost pixel to pixel, so grouping by exact height value
created a boundary (and a new tiny polygon) at nearly every pixel -- a
1024x1024 test window alone produced 5,000-50,000 polygons, and extrapolated
to this script's real window size that was enough to OOM-kill the process.
Collapsing to one canopy class first means two adjacent pixels at different
heights merge into the same region instead of splitting into separate slivers.

The raster is ~320 billion pixels total, far too much to polygonize in one
pass (or even to hold one row's worth of windows in memory at a sensible
window size), so this works in small fixed-size windows (default 8192x8192
~= 268MB per band as uint32, safely small) and is meant to run as a sharded
Batch array job -- see aws-notes/setup-commands.sh for the pattern. Each
shard loops over its windows (common.py's shard_keys works on any list, not
just S3 keys) and writes one combined GeoParquet per shard; merging shards
into one dataset is a separate, cheap final step (just concatenating
GeoParquet files).
"""

import argparse
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely
from osgeo import gdal

from common import S3_BUCKET, add_shard_args, shard_keys, upload_to_s3

gdal.UseExceptions()

RASTER_PATH = "/home/ian/repos/lidar-classify-by-features/canopy-files/raster/ky_canopy_height_3857.tif"
BOUNDARY_PATH = "/home/ian/repos/lidar-classify-by-features/canopy-files/boundary/ky_boundary_buffered_1000ft_3857.gpkg"
RASTER_CRS = "EPSG:3857"
DEFAULT_WINDOW_SIZE = 8192
POLYGONS_PREFIX = "canopy/polygons/"


def compute_windows(raster_xsize, raster_ysize, window_size):
    """(xoff, yoff, xsize, ysize) tuples tiling the whole raster, clipped at the edges."""

    windows = []
    for yoff in range(0, raster_ysize, window_size):
        ysize = min(window_size, raster_ysize - yoff)
        for xoff in range(0, raster_xsize, window_size):
            xsize = min(window_size, raster_xsize - xoff)
            windows.append((xoff, yoff, xsize, ysize))
    return windows


def process_window(raster_path, boundary_geoms, window):
    """
    One window's polygons, or None if nothing in it survives the mask
    (no canopy, or entirely outside the buffered boundary / valid data).
    """

    xoff, yoff, xsize, ysize = window
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    height = band.ReadAsArray(xoff, yoff, xsize, ysize)
    valid = band.GetMaskBand().ReadAsArray(xoff, yoff, xsize, ysize) != 0
    ds = None

    keep = valid & (height > 0)
    if not keep.any():
        return None

    window_transform = rasterio.transform.Affine(
        gt[1], gt[2], gt[0] + xoff * gt[1] + yoff * gt[2],
        gt[4], gt[5], gt[3] + xoff * gt[4] + yoff * gt[5],
    )

    boundary_mask = rasterio.features.rasterize(
        boundary_geoms, out_shape=(ysize, xsize), transform=window_transform,
        fill=0, default_value=1, dtype=np.uint8,
    ).astype(bool)
    keep &= boundary_mask
    if not keep.any():
        return None

    # One class (canopy present), not grouped by height value -- see module
    # docstring for why per-height grouping is impractical here. keep's own
    # dtype (bool) burns True regions as 1, False/outside as 0 or skipped
    # via mask -- only the True shapes come out of shapes() below.
    shapes = rasterio.features.shapes(keep.astype(np.uint8), mask=keep, transform=window_transform, connectivity=8)
    # rasterio.features.shapes yields GeoJSON-like mappings, not shapely geometries.
    geoms = [shapely.geometry.shape(geom) for geom, _value in shapes]
    if not geoms:
        return None

    return gpd.GeoDataFrame({"canopy": True}, index=range(len(geoms)), geometry=geoms, crs=RASTER_CRS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raster", default=RASTER_PATH)
    parser.add_argument("--boundary", default=BOUNDARY_PATH)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--workers", type=int, default=4, help="within-shard window concurrency")
    parser.add_argument("--local-out", help="write the merged shard result here instead of uploading to S3")
    add_shard_args(parser)
    args = parser.parse_args()

    boundary = gpd.read_file(args.boundary)
    boundary_geoms = list(boundary.geometry.values)

    ds = gdal.Open(args.raster)
    raster_xsize, raster_ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    windows = compute_windows(raster_xsize, raster_ysize, args.window_size)
    print(f"{len(windows):,} total windows at {args.window_size}px")

    windows = shard_keys(windows, args.shard_index, args.num_shards)
    if args.shard_index is not None:
        print(f"shard {args.shard_index}/{args.num_shards}: {len(windows)} windows")

    start = time.perf_counter()
    frames = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_window, args.raster, boundary_geoms, w): w for w in windows}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                frames.append(result)
            if done % 200 == 0:
                print(f"{done}/{len(windows)} windows done, {len(frames)} non-empty so far")

    elapsed = time.perf_counter() - start
    if not frames:
        print(f"no canopy polygons in this shard's windows ({elapsed:.1f}s)")
        return

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=RASTER_CRS)
    print(f"{len(merged):,} polygons from {len(frames)} non-empty windows in {elapsed:.1f}s")

    if args.local_out:
        merged.to_parquet(args.local_out)
        print(f"wrote {args.local_out}")
        return

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    merged.to_parquet(tmp_path)
    key = f"{POLYGONS_PREFIX}shard_{args.shard_index}_of_{args.num_shards}.parquet"
    upload_to_s3(tmp_path, S3_BUCKET, key)
    tmp_path.unlink()


if __name__ == "__main__":
    main()
