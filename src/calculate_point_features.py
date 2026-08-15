#!/usr/bin/env python

############################################################
# Calculate per-point geometric/contextual features for a
# lidar tile: HAG, PCA-based geometric features (vectorized),
# and within-building-footprint, exported to Parquet.
# Ian Horn
# August 11, 2026
############################################################

import json
import os
import time
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import pdal

# Opt-in only, via an env var the Coiled branch never sets -- unset here
# means the existing workers=-1 behavior (all cores) is unchanged there.
_KDTREE_WORKERS = int(os.environ.get("LIDAR_LOCAL_MAX_WORKERS", -1))
from scipy.spatial import cKDTree
from shapely import contains_xy

S3_BUCKET = "lidar-classification"


def upload_to_s3(local_path, bucket, key):
    """Upload a local file to S3."""

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded {local_path} to s3://{bucket}/{key}")


def load_points(lasfile):
    """
    Read a LAS file and add HeightAboveGround via filters.hag_delaunay
    (builds a ground TIN from Classification == 2 points, computes each point's
    height above it). HeightAboveGround comes out in the same units as the
    source data (US survey feet here), since it's just Z - interpolated_ground_Z.
    """

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(lasfile)},
            {"type": "filters.hag_delaunay"},
        ]
    }

    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()

    return p.arrays[0]


def compute_geometric_features_vectorized(tree, xyz, radius, min_neighbors=5, chunk_size=300_000):
    """
    Batched, vectorized PCA-based geometric features over a radius search:
    Planarity, Linearity, Sphericity, SurfaceVariation, Roughness, Verticality,
    Normal(X/Y/Z), NeighborCount, Density. ~19x faster on a 12.3M point cloud
    than a per-point loop calling query_ball_point/np.cov/np.linalg.eigh
    individually (verified against the loop version: max abs difference ~1e-15).

    radius is in the point cloud's native CRS units (US survey feet for
    NAD83 / Kentucky Single Zone ftUS). Swept radius=4/5/7/10ft on this
    dataset; 7ft gives 99.5% neighbor coverage with stable neighbor counts,
    without going as far as 10ft where Roughness starts picking up
    whole-building-scale structure instead of local surface texture.

    Processed in chunks because the full flattened (point, neighbor) pair
    list doesn't comfortably fit in memory at once for a full tile.
    """

    n_points = len(xyz)
    sphere_volume = (4.0 / 3.0) * np.pi * radius ** 3

    out = {
        "Planarity": np.zeros(n_points),
        "Linearity": np.zeros(n_points),
        "Sphericity": np.zeros(n_points),
        "SurfaceVariation": np.zeros(n_points),
        "Roughness": np.zeros(n_points),
        "Verticality": np.zeros(n_points),
        "NormalX": np.zeros(n_points),
        "NormalY": np.zeros(n_points),
        "NormalZ": np.zeros(n_points),
        "NeighborCount": np.zeros(n_points, dtype=np.int64),
        "Density": np.zeros(n_points),
    }

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        chunk_idx = np.arange(start, stop)
        chunk_xyz = xyz[chunk_idx]
        n_chunk = len(chunk_idx)

        # Batched, parallel radius search for the whole chunk
        neighbor_lists = tree.query_ball_point(chunk_xyz, r=radius, workers=_KDTREE_WORKERS)
        lengths = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64, count=n_chunk)
        valid = lengths >= min_neighbors

        # Density is defined for every point regardless of the PCA min_neighbors floor --
        # it's just neighbor count over the search-sphere volume, points per cubic foot
        out["NeighborCount"][chunk_idx] = lengths
        out["Density"][chunk_idx] = lengths / sphere_volume

        # Flatten (point, neighbor) pairs
        point_ids = np.repeat(np.arange(n_chunk), lengths)
        neighbor_ids = np.concatenate(neighbor_lists) if lengths.sum() else np.empty(0, dtype=np.int64)
        neighbor_xyz = xyz[neighbor_ids]

        # Centroid per point via segment sum
        safe_lengths = np.maximum(lengths, 1)
        sums = np.zeros((n_chunk, 3))
        np.add.at(sums, point_ids, neighbor_xyz)
        centroid = sums / safe_lengths[:, None]

        centered = neighbor_xyz - centroid[point_ids]

        # Covariance matrix per point, accumulated with bincount instead of per-point np.cov
        cxx = np.bincount(point_ids, weights=centered[:, 0] * centered[:, 0], minlength=n_chunk)
        cxy = np.bincount(point_ids, weights=centered[:, 0] * centered[:, 1], minlength=n_chunk)
        cxz = np.bincount(point_ids, weights=centered[:, 0] * centered[:, 2], minlength=n_chunk)
        cyy = np.bincount(point_ids, weights=centered[:, 1] * centered[:, 1], minlength=n_chunk)
        cyz = np.bincount(point_ids, weights=centered[:, 1] * centered[:, 2], minlength=n_chunk)
        czz = np.bincount(point_ids, weights=centered[:, 2] * centered[:, 2], minlength=n_chunk)

        denom = np.maximum(lengths - 1, 1)  # ddof=1, matches np.cov default
        cov = np.zeros((n_chunk, 3, 3))
        cov[:, 0, 0] = cxx / denom
        cov[:, 1, 1] = cyy / denom
        cov[:, 2, 2] = czz / denom
        cov[:, 0, 1] = cov[:, 1, 0] = cxy / denom
        cov[:, 0, 2] = cov[:, 2, 0] = cxz / denom
        cov[:, 1, 2] = cov[:, 2, 1] = cyz / denom

        # Batched eigendecomposition of the whole chunk's covariance matrices at once.
        # eigh returns ascending eigenvalues: index 0 = smallest (l3), index 2 = largest (l1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        l3, l2, l1 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
        normal = eigvecs[:, :, 0]

        valid = valid & (l1 > 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            linearity = np.where(valid, (l1 - l2) / l1, 0.0)
            planarity = np.where(valid, (l2 - l3) / l1, 0.0)
            sphericity = np.where(valid, l3 / l1, 0.0)
            total = l1 + l2 + l3
            surface_variation = np.where(valid & (total > 0), l3 / total, 0.0)

        # Roughness = std of distance to best-fit plane, via segment sum/sum-of-squares
        dist = np.einsum("ij,ij->i", centered, normal[point_ids])
        sum_d = np.bincount(point_ids, weights=dist, minlength=n_chunk)
        sum_d2 = np.bincount(point_ids, weights=dist ** 2, minlength=n_chunk)
        mean_d = sum_d / safe_lengths
        roughness = np.sqrt(np.maximum(sum_d2 / safe_lengths - mean_d ** 2, 0.0))
        roughness = np.where(valid, roughness, 0.0)

        # Verticality: 0 for a horizontal surface (normal ~ +-Z), 1 for a vertical
        # surface (normal perpendicular to Z) -- same definition PDAL's
        # filters.covariancefeatures uses.
        verticality = np.where(valid, 1.0 - np.abs(normal[:, 2]), 0.0)

        normal = np.where(valid[:, None], normal, 0.0)

        out["Planarity"][chunk_idx] = planarity
        out["Linearity"][chunk_idx] = linearity
        out["Sphericity"][chunk_idx] = sphericity
        out["SurfaceVariation"][chunk_idx] = surface_variation
        out["Roughness"][chunk_idx] = roughness
        out["Verticality"][chunk_idx] = verticality
        out["NormalX"][chunk_idx] = normal[:, 0]
        out["NormalY"][chunk_idx] = normal[:, 1]
        out["NormalZ"][chunk_idx] = normal[:, 2]

    return out


def compute_within_footprint(xyz, footprints_file, crs=3089):
    """
    Point-in-polygon test against building footprints, with a per-building
    bbox prefilter before the exact contains_xy check -- looping the exact
    test over every point x every building would be wasteful, so the cheap
    bbox filter cuts the candidate set down first.

    This is a candidate/prior signal, not a building label: footprints come
    from a later Overture release than the lidar capture date, so a footprint
    can exist for something built after the lidar flight (points there are
    ground/site work, not a building), and a building present at capture time
    can have since been demolished or remodeled and no longer match the
    current footprint's shape or height. Combine with HAG and the geometric
    features downstream rather than treating this as ground truth.
    """

    buildings_gdf = gpd.read_parquet(footprints_file)
    buildings_gdf = buildings_gdf.to_crs(crs)  # match lidar CRS

    x, y = xyz[:, 0], xyz[:, 1]
    within_footprint = np.zeros(len(xyz), dtype=bool)
    footprint_height = np.full(len(xyz), np.nan)

    for height, geom in zip(buildings_gdf["height"], buildings_gdf.geometry):
        minx, miny, maxx, maxy = geom.bounds
        candidate = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        if not candidate.any():
            continue
        inside = np.zeros(len(xyz), dtype=bool)
        inside[candidate] = contains_xy(geom, x[candidate], y[candidate])
        within_footprint |= inside
        footprint_height[inside] = height

    return within_footprint, footprint_height


def export_features(xyz, points, hag, geom_vec, output_file):
    """Assemble one row per point and write to Parquet (pandas/pyarrow, no GDAL dependency)."""

    features_df = pd.DataFrame({
        "X": xyz[:, 0],
        "Y": xyz[:, 1],
        "Z": xyz[:, 2],
        "Intensity": points["Intensity"],
        "Classification": points["Classification"],
        "HAG": hag,
        **geom_vec,
    })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_file)

    print(f"wrote {len(features_df):,} rows x {len(features_df.columns)} columns to {output_file}")

    upload_to_s3(output_file, S3_BUCKET, f"phase2/features/{Path(output_file).name}")

    # This is the terminal output of the pipeline for this tile -- nothing
    # downstream reads the local copy once it's in S3, so clear it rather
    # than accumulating per-tile parquet files on local/environment disk.
    Path(output_file).unlink()

    return features_df


def main():

    lasfile = "lidar-files/N075E299.laz"
    footprints_file = "building-files/N075E299.parquet"
    output_file = "lidar-features/N075E299.parquet"
    radius = 7.0
    chunk_size = 90_000

    points = load_points(lasfile)
    hag = points["HeightAboveGround"]

    xyz = np.column_stack([points["X"], points["Y"], points["Z"]])
    tree = cKDTree(xyz)
    print(f"{len(xyz):,} points loaded")

    start = time.perf_counter()
    geom_vec = compute_geometric_features_vectorized(tree, xyz, radius=radius, chunk_size=chunk_size)
    print(f"geometric features: {time.perf_counter() - start:.1f}s")

    within_footprint, footprint_height = compute_within_footprint(xyz, footprints_file)
    geom_vec["WithinFootprint"] = within_footprint
    geom_vec["FootprintHeight"] = footprint_height

    export_features(xyz, points, hag, geom_vec, output_file)


if __name__ == "__main__":
    main()
