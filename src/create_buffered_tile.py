#!/usr/bin/env python3

"""
This script uses the bounding box of kyfromabove lidar tile, buffers it by a
specified distance, uses the buffered bounding box to search STAC for lidar tiles.
The points from each lidar are read to create a new temporary local tile.
"""

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote

try:
    import pdal
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    pdal = None

try:
    import requests
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    requests = None

try:
    from pystac import Item
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    Item = None

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    Transformer = None

try:
    from pystac_client import Client
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    Client = None

try:
    import duckdb
    import geopandas as gpd
    import pyarrow.parquet as pq
    from shapely import wkb
    from shapely.geometry import box
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    duckdb = None
    gpd = None
    pq = None
    wkb = None
    box = None


def ensure_runtime_dependencies():
    """Install Python packages needed by the buffering and vector-export workflow."""

    global duckdb, gpd, pq, wkb, box, Item, Transformer, Client, requests

    project_root = Path(__file__).resolve().parents[1]
    venv_dir = project_root / ".venv"
    if os.name == "nt":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"

    if not python_exe.exists():
        subprocess.run(["uv", "venv", "--seed", str(venv_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    packages = ["requests", "pystac", "pystac-client", "pyproj"]
    if duckdb is None or gpd is None or wkb is None:
        packages.extend(["duckdb", "geopandas", "pyarrow", "shapely"])

    try:
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python_exe), *packages],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install runtime dependencies: {exc.stdout}") from exc

    site_packages = subprocess.check_output(
        [str(python_exe), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    if site_packages and site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    if duckdb is None or gpd is None or pq is None or wkb is None or box is None:
        try:
            import duckdb as duckdb_module
            import geopandas as geopandas_module
            import pyarrow.parquet as parquet_module
            from shapely import wkb as wkb_module
            from shapely.geometry import box as box_module
        except ImportError as exc:
            raise RuntimeError(f"Vector dependencies are still unavailable after bootstrap: {exc}") from exc

        duckdb = duckdb_module
        gpd = geopandas_module
        pq = parquet_module
        wkb = wkb_module
        box = box_module

    try:
        import requests as requests_module
        from pystac import Item as item_class
        from pyproj import Transformer as transformer_class
        from pystac_client import Client as client_class
    except ImportError as exc:
        raise RuntimeError(f"Runtime dependencies are still unavailable after bootstrap: {exc}") from exc

    requests = requests_module
    Item = item_class
    Transformer = transformer_class
    Client = client_class


def ensure_vector_dependencies():
    """Install the optional vector stack into a local virtualenv when it is missing."""

    ensure_runtime_dependencies()
    if gpd is None or pq is None or wkb is None or box is None:
        raise RuntimeError("geopandas, pyarrow, and shapely are required to export Overture building footprints")


def get_stac_item(item_id: str, item_collection: str, item_api_url: str):
    """
    this function uses pystac to grab the item from file (href)
    """
    
    href = f'{item_api_url}/collections/{item_collection}/items/{item_id}'
    stac_item = Item.from_file(href)
    return stac_item


def get_distance_degrees(meters: float):
    """
    This functions calculates the buffer distance in meters
    and converts to degrees because we are working with 
    WGS84.
    """

    meters_per_degree = 111_320
    distance_meters = meters
    distance_degrees = distance_meters / meters_per_degree
    return distance_degrees


def get_buffered_bbox(item_bbox, distance_degrees):
    """
    We need to create a buffered bbox so we can later 
    query the stac to include surrounding tiles.
    """

    minx, miny, maxx, maxy = item_bbox

    buffered_bbox = [
        minx - distance_degrees,
        miny - distance_degrees,
        maxx + distance_degrees,
        maxy + distance_degrees
    ]
    return buffered_bbox


def reproject_bbox(bbox, src_epsg=4326, dst_epsg=3089):
    """
    Reproject a WGS84 bbox to Kentucky Single Zone (EPSG:3089).
    """

    transformer = Transformer.from_crs(
        src_epsg,
        dst_epsg,
        always_xy=True
    )

    minx, miny, maxx, maxy = bbox

    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(maxx, maxy)

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2)
    ]


def pdal_bounds(bbox):
    xmin, ymin, xmax, ymax = bbox
    return f"([{xmin},{xmax}],[{ymin},{ymax}])"

def search_stac(stac_api: str, collection: str, buffered_bbox):
    """
    Use pystac_client to open the stac api
    search by bbox
    return a list of hrefs
    """

    client = Client.open(f'{stac_api}/')
    search = client.search(
        max_items=10,
        collections=collection,
        bbox = buffered_bbox
    )

    print(f'Found {len(list(search.items()))} items\n')
    item_list = list(search.items())
    for i in item_list:
        print(i)

    href_list = []

    for item in item_list:
        for asset in item.assets.values():
            href_list.append(asset.href)
    for h in href_list:
        print(h)
        
    return href_list


def get_vector_geometries_dir(output_laz: Path) -> Path:
    return output_laz.parent / "vector_geometries"


def export_overture_buildings(bbox, output_parquet: Path, release: str = "2025-03-19.0-beta.0") -> Path:
    """Download Overture building footprints that intersect a WGS84 bounding box."""

    ensure_vector_dependencies()
    if gpd is None or pq is None or wkb is None or box is None:
        raise RuntimeError(
            "geopandas, pyarrow, and shapely are required to export Overture building footprints"
        )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = bbox
    envelope = box(minx, miny, maxx, maxy)

    prefix = (
        f"bridgefiles/{release}/dataset=Esri Community Maps/theme=buildings/type=building/"
    )
    listing_url = f"https://overturemaps-us-west-2.s3.amazonaws.com/?prefix={quote(prefix)}"
    response = requests.get(listing_url, timeout=120)
    response.raise_for_status()

    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(response.text)
    keys = []
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", namespaces=namespace)
        if key and key.endswith(".parquet"):
            keys.append(key)

    if not keys:
        raise RuntimeError(f"No Overture building parquet files were found for prefix {prefix}")

    records = []
    for key in keys:
        object_url = f"https://overturemaps-us-west-2.s3.amazonaws.com/{quote(key, safe='/')}"
        parquet_response = requests.get(object_url, timeout=120)
        parquet_response.raise_for_status()

        with BytesIO(parquet_response.content) as handle:
            table = pq.read_table(handle)

        if len(table) == 0:
            continue

        frame = table.to_pandas()
        if "geometry" not in frame.columns:
            continue

        for row in frame.itertuples(index=False):
            geometry = getattr(row, "geometry")
            if geometry is None:
                continue

            parsed_geometry = None
            if isinstance(geometry, (bytes, bytearray)):
                try:
                    parsed_geometry = wkb.loads(geometry)
                except Exception:
                    parsed_geometry = None
            elif isinstance(geometry, str):
                try:
                    parsed_geometry = wkb.loads(geometry)
                except Exception:
                    parsed_geometry = None

            if parsed_geometry is None or parsed_geometry.is_empty:
                continue
            if not parsed_geometry.intersects(envelope):
                continue

            record = {
                "id": getattr(row, "id", None),
                "class": getattr(row, "class", None),
                "subclass": getattr(row, "subclass", None),
                "confidence": getattr(row, "confidence", None),
                "names": getattr(row, "names", None),
                "sources": getattr(row, "sources", None),
                "geometry": parsed_geometry,
            }
            records.append(record)

    if not records:
        gdf = gpd.GeoDataFrame(
            {
                "id": [],
                "class": [],
                "subclass": [],
                "confidence": [],
                "names": [],
                "sources": [],
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
    else:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    gdf.to_parquet(output_parquet, index=False)
    return output_parquet


def crop_copc(hrefs, bounds, out_laz): 

    if pdal is None:
        raise RuntimeError("pdal is required to create the merged LAS/LAZ tile")

    pipeline = []

    for href in hrefs:
        pipeline.append(
            {
                "type": "readers.copc",
                "filename": href,
                "bounds": bounds
            }
        )

    pipeline.append(
        {
            "type": "writers.las",
            "filename": str(out_laz)
        }
    )

    print(json.dumps(pipeline, indent=2))

    p = pdal.Pipeline(json.dumps(pipeline))

    start = time.perf_counter()
    count = p.execute()
    elapsed = time.perf_counter() - start

    print(f"{count:,} points written")
    print(f"Elapsed: {elapsed:.1f} seconds")

def main():

    item_id = 'N075E299_LAS_Phase2.copc'
    out_laz = Path('/mnt/d/Data/lidar/N075E299.laz')
    # item_id = 'N075E295_LAS_Phase2.copc'
    collection = 'laz-phase2'
    stac = 'https://spved5ihrl.execute-api.us-west-2.amazonaws.com'

    item = get_stac_item(item_id, collection, stac)
    bbox = item.bbox
    print(f'bbox {bbox}') 

    buffer = 30
    distance = get_distance_degrees(buffer)
    print(f'Degrees: {distance}')
    
    bbox_buffer = get_buffered_bbox(item.bbox, distance)
    print(f'Buffered bbox: {bbox_buffer}')

    vector_geometries_dir = get_vector_geometries_dir(out_laz)
    footprint_parquet = vector_geometries_dir / f'{out_laz.stem}_buildings.parquet'
    print(f'Exporting Overture building footprints to {footprint_parquet}')
    export_overture_buildings(bbox_buffer, footprint_parquet)

    hrefs = search_stac(stac, collection, bbox_buffer)
    # print(stac_search)
    for h in hrefs:
        print(h)

    bbox_3089 = reproject_bbox(bbox_buffer)

    print(f"3089 bbox: {bbox_3089}")

    bounds = pdal_bounds(bbox_3089)
    crop_copc(hrefs, bounds, out_laz)


if __name__ == '__main__':
    main()
