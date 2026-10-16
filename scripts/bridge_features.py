#!/usr/bin/env python3

"""
Bridge-proximity features from the KYTC bridge inventory (BridgeLocationsLine
shapefile): DistToBridge and InBridgeBuffer, computed from each point's X/Y.

Run directly on one features parquet to prototype/inspect a tile:

    python bridge_features.py /path/to/tile.parquet --out-dir /mnt/d/Data/bridge_test
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

LINES_SHP = "/mnt/d/Data/bridge_test/BridgeLocationsLine/BridgeLocationsLine.shp"
LIDAR_CRS = 3089
MAX_DIST_FT = 200.0
DENSIFY_FT = 5.0
BUFFER_MARGIN_FT = 0.0
LANE_FT = 12.0
SHOULDER_FT = 8.0
MAX_ROUTE_WIDTH_FT = 100.0


def load_bridges(path=LINES_SHP):
    """Bridge lines in the lidar CRS with a per-bridge buffer distance (half deck width + margin)."""

    gdf = gpd.read_file(path)

    # CULVRATING is 'N' when the structure is not a culvert; culverts are buried
    # in the ground and can't be confused with a building or elevated deck.
    gdf = gdf[gdf["CULVRATING"] == "N"].copy()

    # The inventory also has one record per route passing *under* a structure
    # ("One Route Under", "2nd Route Under", ...), sharing the bridge's ID and
    # LENGTH; those lines follow the road beneath the bridge, not the deck.
    gdf = gdf[~gdf["D_ON_UNDER"].str.contains("Under", na=False)].copy()

    # The shapefile is EPSG:6473 (NAD83(2011)); the tile coordinates are 3089.
    gdf = gdf.to_crs(LIDAR_CRS)

    # DECKWIDTH is the whole structure's width, repeated on every route record
    # (mainline, ramps, cross street), while each line is only one route's
    # centerline -- so buffering by DECKWIDTH draws the full structure around
    # every route. Use this route's own width instead: ROADWIDTH when given,
    # otherwise LANES-based, never wider than the structure itself.
    lane_width = gdf["LANES"] * LANE_FT + SHOULDER_FT
    route_width = gdf["ROADWIDTH"].where(gdf["ROADWIDTH"] > 0, lane_width).clip(upper=MAX_ROUTE_WIDTH_FT)
    route_width = route_width.where(~(gdf["DECKWIDTH"] > 0), np.minimum(route_width, gdf["DECKWIDTH"]))
    gdf["BufferFt"] = route_width / 2 + BUFFER_MARGIN_FT

    return gdf.reset_index(drop=True)


def bridges_near_points(bridges, x, y):
    """Bridges whose lines come within MAX_DIST_FT of the points' bounding box."""

    search_box = shapely.box(x.min(), y.min(), x.max(), y.max()).buffer(MAX_DIST_FT)
    return bridges[bridges.intersects(search_box)].reset_index(drop=True)


def compute_bridge_features(x, y, bridges):
    """
    DistToBridge: distance (ft) to the nearest bridge line, capped at MAX_DIST_FT.
    InBridgeBuffer: True if the point is inside its bridge's deck-width buffer.

    Distance is measured to the nearest vertex of the lines densified every
    DENSIFY_FT, so it can read up to DENSIFY_FT/2 high -- fine at this scale
    and it lets one cKDTree query cover every point.
    """

    vertices = shapely.get_coordinates(shapely.segmentize(bridges.geometry.values, DENSIFY_FT))
    tree = cKDTree(vertices)
    dist, _ = tree.query(np.column_stack([x, y]), distance_upper_bound=MAX_DIST_FT, workers=-1)
    dist = np.where(np.isfinite(dist), dist, MAX_DIST_FT)

    buffers = shapely.union_all(shapely.buffer(bridges.geometry.values, bridges["BufferFt"].values, cap_style="flat"))
    shapely.prepare(buffers)

    # Only points near a line can be inside a buffer, so skip the exact test elsewhere.
    candidate = dist <= bridges["BufferFt"].max()
    in_buffer = np.zeros(len(x), dtype=bool)
    in_buffer[candidate] = shapely.contains_xy(buffers, x[candidate], y[candidate])

    return dist.astype(np.float32), in_buffer


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

    bridges = load_bridges()
    tile_bridges = bridges_near_points(bridges, x, y)
    print(f"{len(bridges):,} non-culvert bridges statewide, {len(tile_bridges)} on this tile")

    start = time.perf_counter()
    dist, in_buffer = compute_bridge_features(x, y, tile_bridges)
    print(f"bridge features: {time.perf_counter() - start:.1f}s")

    df["DistToBridge"] = dist
    df["InBridgeBuffer"] = in_buffer

    print(f"InBridgeBuffer: {in_buffer.sum():,} points ({in_buffer.mean():.2%})")
    print("\nHAG (ft) percentiles, in buffer vs. elsewhere:")
    print(pd.DataFrame({
        "in_buffer": df.loc[in_buffer, "HAG"].describe(percentiles=[0.1, 0.5, 0.9]),
        "elsewhere": df.loc[~in_buffer, "HAG"].describe(percentiles=[0.1, 0.5, 0.9]),
    }).round(1))
    print("\nClassification counts inside buffers:")
    print(df.loc[in_buffer, "Classification"].value_counts().head(8))
    print(f"\nin buffer AND WithinFootprint (building footprint over a bridge): {(in_buffer & df['WithinFootprint']).sum():,}")

    stem = Path(args.parquet).name.split("_")[0]
    buffer_polys = tile_bridges.copy()
    buffer_polys["geometry"] = shapely.buffer(tile_bridges.geometry.values, tile_bridges["BufferFt"].values, cap_style="flat")
    buffer_polys[["BRIDGE_ID", "FEATINT", "DECKWIDTH", "LENGTH", "SERVTYPUND", "BufferFt", "geometry"]].to_file(
        out_dir / f"{stem}_bridge_buffers.gpkg", driver="GPKG"
    )

    near = df[df["DistToBridge"] < 60]
    near_gdf = gpd.GeoDataFrame(
        near.drop(columns=["X", "Y", "Z"]),
        geometry=shapely.points(near["X"].to_numpy(), near["Y"].to_numpy(), near["Z"].to_numpy()),
        crs=LIDAR_CRS,
    )
    near_gdf.to_file(out_dir / f"{stem}_points_near_bridges.gpkg", driver="GPKG")
    print(f"\nwrote buffers and {len(near_gdf):,} points within 60ft of a bridge to {out_dir}")


if __name__ == "__main__":
    main()
