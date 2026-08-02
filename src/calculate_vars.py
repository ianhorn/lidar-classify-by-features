#!/usr/bin/env python3
"""Compute point-level geometric features from a LAS/LAZ point cloud.

The script uses PDAL to derive height above ground (HAG) and surface normals,
then adds local geometric features such as planarity, roughness, density, and
optional inside-footprint membership.

The resulting CSV can be used as training data for a classifier that will first
separate building points from other classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from shapely.geometry import MultiPolygon, Point, Polygon, shape
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    MultiPolygon = Polygon = Point = None
    shape = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate geometric features for LAS/LAZ points")
    parser.add_argument("input_las", type=Path, help="Path to the input LAS/LAZ file")
    parser.add_argument("output_csv", type=Path, help="Path to the output CSV file")
    parser.add_argument(
        "--footprint-file",
        type=Path,
        help="Optional GeoJSON file containing building footprints (Polygon/MultiPolygon)",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=0.5,
        help="Radius in the point cloud's native coordinates used for local feature estimation (default: 0.5)",
    )
    parser.add_argument(
        "--pdal-executable",
        default="pdal",
        help="PDAL executable to invoke (default: pdal)",
    )
    parser.add_argument(
        "--point-limit",
        type=int,
        default=None,
        help="Optional limit for the number of points to process (useful for testing)",
    )
    return parser.parse_args()


def ensure_pdal_available(executable: str) -> None:
    try:
        subprocess.run([executable, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"PDAL executable '{executable}' was not found. Install PDAL and ensure it is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"PDAL executable '{executable}' is present but did not respond as expected: {exc.stderr.decode('utf-8', 'ignore')}"
        ) from exc


def build_pdal_pipeline(input_las: Path, output_csv: Path) -> Dict[str, Any]:
    return {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {"type": "filters.hag_delaunay"},
            {"type": "filters.normal", "knn": 10},
            {
                "type": "writers.text",
                "filename": str(output_csv),
                "order": ["X", "Y", "Z", "HeightAboveGround", "NormalX", "NormalY", "NormalZ"],
            },
        ]
    }


def run_pdal_pipeline(input_las: Path, output_csv: Path, executable: str) -> Path:
    ensure_pdal_available(executable)
    pipeline = build_pdal_pipeline(input_las, output_csv)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(pipeline, handle, indent=2)
        temp_pipeline_path = Path(handle.name)

    try:
        result = subprocess.run(
            [executable, "pipeline", str(temp_pipeline_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
    finally:
        temp_pipeline_path.unlink(missing_ok=True)

    return output_csv


def make_record_lookup(record: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in record.items():
        normalized[normalize_key(key)] = value
    return normalized


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def parse_pdal_output(output_csv: Path) -> List[Dict[str, Any]]:
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    header = [cell.strip() for cell in rows[0]]
    records: List[Dict[str, Any]] = []
    for row in rows[1:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            row = row[: len(header)] + [""] * (len(header) - len(row))
        records.append({header[idx]: row[idx] for idx in range(len(header))})
    return records


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_footprint(path: Path):
    if shape is None or Point is None:
        raise RuntimeError("shapely is required to use --footprint-file")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("type") in {"Polygon", "MultiPolygon"}:
        geometry = shape(payload)
    else:
        geometries = []
        for feature in payload.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                geometries.append(shape(geometry))
        if not geometries:
            raise ValueError(f"No polygonal geometries were found in {path}")
        geometry = geometries[0]
        if len(geometries) > 1:
            from shapely.ops import unary_union

            geometry = unary_union(geometries)

    if geometry.is_empty:
        raise ValueError(f"The footprint in {path} was empty")
    return geometry


def fit_plane(points: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if len(points) < 3:
        return (0.0, 0.0, 0.0)

    n = len(points)
    sxx = sum(x * x for x, _, _ in points)
    syy = sum(y * y for _, y, _ in points)
    sxy = sum(x * y for x, y, _ in points)
    sx = sum(x for x, _, _ in points)
    sy = sum(y for _, y, _ in points)
    sz = sum(z for _, _, z in points)
    sxz = sum(x * z for x, _, z in points)
    syz = sum(y * z for _, y, z in points)

    matrix = [
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx, sy, float(n)],
    ]
    rhs = [sxz, syz, sz]

    # Gaussian elimination with partial pivoting.
    a = [row[:] for row in matrix]
    b = rhs[:]

    for i in range(3):
        pivot = max(range(i, 3), key=lambda row: abs(a[row][i]))
        if abs(a[pivot][i]) < 1e-12:
            return (0.0, 0.0, 0.0)
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]

        for row in range(i + 1, 3):
            factor = a[row][i] / a[i][i]
            if factor == 0.0:
                continue
            for col in range(i, 3):
                a[row][col] -= factor * a[i][col]
            b[row] -= factor * b[i]

    x_vals = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        total = b[row]
        for col in range(row + 1, 3):
            total -= a[row][col] * x_vals[col]
        x_vals[row] = total / a[row][row]

    return (x_vals[0], x_vals[1], x_vals[2])


def plane_value(a: Tuple[float, float, float], x: float, y: float) -> float:
    slope_x, slope_y, intercept = a
    return slope_x * x + slope_y * y + intercept


def build_spatial_index(points: Sequence[Tuple[float, float, float]], cell_size: float) -> Dict[Tuple[int, int], List[int]]:
    index: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y, _) in enumerate(points):
        cell = (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
        index.setdefault(cell, []).append(idx)
    return index


def collect_neighbors(index: Dict[Tuple[int, int], List[int]], points: Sequence[Tuple[float, float, float]], idx: int, radius: float) -> List[Tuple[float, float, float]]:
    x0, y0, _ = points[idx]
    radius_sq = radius * radius
    neighbors: List[Tuple[float, float, float]] = []
    cell = (int(math.floor(x0 / radius)), int(math.floor(y0 / radius)))

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for candidate_idx in index.get((cell[0] + dx, cell[1] + dy), []):
                if candidate_idx == idx:
                    continue
                x1, y1, z1 = points[candidate_idx]
                dx_val = x1 - x0
                dy_val = y1 - y0
                if dx_val * dx_val + dy_val * dy_val > radius_sq:
                    continue
                neighbors.append((x1, y1, z1))

    return neighbors


def compute_local_geometry(points: Sequence[Tuple[float, float, float]], radius: float) -> List[Tuple[Optional[float], Optional[float], Optional[float]]]:
    if not points:
        return []

    cell_size = max(radius, 1e-6)
    index = build_spatial_index(points, cell_size)
    results: List[Tuple[Optional[float], Optional[float], Optional[float]]] = []

    for idx, (x, y, z) in enumerate(points):
        neighbors = collect_neighbors(index, points, idx, radius)
        if len(neighbors) < 3:
            results.append((None, None, None))
            continue

        neighbor_points = [(x, y, z)] + neighbors
        plane = fit_plane(neighbor_points)
        residuals = [z_i - plane_value(plane, x_i, y_i) for x_i, y_i, z_i in neighbor_points]
        residual_mean = sum(residuals) / len(residuals)
        residual_variance = sum((residual - residual_mean) ** 2 for residual in residuals) / len(residuals)
        roughness = math.sqrt(residual_variance)

        z_range = max(z_i for _, _, z_i in neighbor_points) - min(z_i for _, _, z_i in neighbor_points)
        if z_range <= 1e-9:
            planarity = 1.0
        else:
            planarity = max(0.0, min(1.0, 1.0 - (roughness / z_range)))

        results.append((planarity, roughness, float(len(neighbors))))

    return results


def enrich_points(
    records: Sequence[Dict[str, Any]],
    footprint=None,
    neighbor_radius: float = 0.5,
    point_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if point_limit is not None:
        records = list(records[:point_limit])

    points: List[Tuple[float, float, float]] = []
    parsed_records: List[Dict[str, Any]] = []

    for record in records:
        lookup = make_record_lookup(record)
        x = parse_float(lookup.get("x"))
        y = parse_float(lookup.get("y"))
        z = parse_float(lookup.get("z"))
        if x is None or y is None or z is None:
            continue

        points.append((x, y, z))
        parsed_records.append(lookup)

    local_geometry = compute_local_geometry(points, neighbor_radius)
    enriched_records: List[Dict[str, Any]] = []

    for idx, lookup in enumerate(parsed_records):
        x = parse_float(lookup.get("x"))
        y = parse_float(lookup.get("y"))
        z = parse_float(lookup.get("z"))
        haq = parse_float(lookup.get("heightaboveground"))
        if haq is None:
            haq = parse_float(lookup.get("hag"))
        if haq is None:
            haq = parse_float(lookup.get("height_above_ground"))

        normal_x = parse_float(lookup.get("normalx"))
        normal_y = parse_float(lookup.get("normaly"))
        normal_z = parse_float(lookup.get("normalz"))
        if normal_x is None or normal_y is None or normal_z is None:
            normal_x = parse_float(lookup.get("nx"))
            normal_y = parse_float(lookup.get("ny"))
            normal_z = parse_float(lookup.get("nz"))

        planarity, roughness, density = local_geometry[idx]
        verticality = None
        if normal_z is not None:
            verticality = max(0.0, min(1.0, 1.0 - abs(normal_z)))

        inside_footprint = None
        if footprint is not None and x is not None and y is not None:
            point_geom = Point(x, y)
            inside_footprint = bool(footprint.covers(point_geom))

        enriched_records.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "hag": haq,
                "planarity": planarity,
                "roughness": roughness,
                "density": density,
                "verticality": verticality,
                "normal_x": normal_x,
                "normal_y": normal_y,
                "normal_z": normal_z,
                "inside_footprint": inside_footprint,
            }
        )

    return enriched_records


def write_output_csv(rows: Sequence[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "x",
        "y",
        "z",
        "hag",
        "planarity",
        "roughness",
        "density",
        "verticality",
        "normal_x",
        "normal_y",
        "normal_z",
        "inside_footprint",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    input_las = args.input_las.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not input_las.exists():
        raise FileNotFoundError(f"Input LAS/LAZ file does not exist: {input_las}")

    footprint = None
    if args.footprint_file is not None:
        footprint_path = args.footprint_file.expanduser().resolve()
        if not footprint_path.exists():
            raise FileNotFoundError(f"Footprint file does not exist: {footprint_path}")
        footprint = load_footprint(footprint_path)

    with tempfile.TemporaryDirectory(prefix="calculate-vars-", dir=str(output_csv.parent)) as temp_dir:
        pdal_output = Path(temp_dir) / "pdal-output.csv"
        run_pdal_pipeline(input_las, pdal_output, args.pdal_executable)
        records = parse_pdal_output(pdal_output)

    enriched_rows = enrich_points(
        records,
        footprint=footprint,
        neighbor_radius=args.neighbor_radius,
        point_limit=args.point_limit,
    )
    write_output_csv(enriched_rows, output_csv)
    print(f"Wrote {len(enriched_rows):,} points to {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI wrapper
        print(f"Error: {exc}", file=sys.stderr)
        raise
