#!/usr/bin/env python3

"""
This script uses the bounding box of kyfromabove lidar tile, buffers it by a 
specfied distance, uses the bufferred bounding box to search stac for lidar tiles.
The points from each lidar will be read to create a new temporary local tile. 
"""


import json
import math
import os
import time
import boto3
import duckdb
import requests
import subprocess
import tempfile
import geopandas as gpd
import pandas as pd
from datetime import datetime, timezone
from pystac import Item
from pathlib import Path
from pystac_client import Client
from concurrent.futures import ThreadPoolExecutor


S3_BUCKET = "lidar-classification"


def upload_to_s3(local_path, bucket, key):
    """Upload a local file to S3. Leaves the local copy in place -- it's still
    needed by downstream steps (feature calculation) in the same environment."""

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"uploaded {local_path} to s3://{bucket}/{key}")


def quarantine_href(href, bucket, prefix="phase2/laz-problematic"):
    """
    Download a COPC file PDAL couldn't read and copy it to S3 for review,
    instead of silently dropping it. Wrapped by the caller so a file that's
    also unreachable by plain HTTP (not just unreadable by PDAL) doesn't
    take down the rest of the crop.
    """

    filename = href.rsplit("/", 1)[-1]
    tmp_path = Path(tempfile.gettempdir()) / filename

    with requests.get(href, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

    upload_to_s3(tmp_path, bucket, f"{prefix}/{filename}")
    tmp_path.unlink()


def crop_single_href(href, bounds, out_path, retries=3, backoff=2.0):
    """
    Crop one COPC file via the `pdal` CLI in a subprocess, not the Python
    bindings. PDAL's Python bindings raise an uncatchable C++ exception on
    some S3 read failures (`terminate called ... Aborted` -- kills the whole
    process, no Python `except:` can catch it); the CLI's own error handling
    exits cleanly with a non-zero return code instead, and subprocess
    isolation means even a hard crash there only kills the subprocess.

    Retries a few times with a short linear backoff before giving up --
    without this, a single transient blip (a brief DNS/network hiccup, a
    momentary S3 throttle) permanently quarantines a perfectly good source
    file and silently undercounts the tile, since `crop_copc` only fails
    hard if every href fails; a partial read failure otherwise never
    surfaces as an error.

    readers.copc needs a well-formed COPC VLR (the octree/spatial-index
    structure) to do its bounded, indexed read -- a source file that's
    mislabeled or otherwise not a real COPC despite its extension fails
    there with a VLR-related error. On that specific failure, switch to
    readers.las + filters.crop for the remaining attempts: readers.las
    reads the file as plain LAS/LAZ with no COPC VLR requirement at all,
    at the cost of a full linear read instead of COPC's indexed one.
    """

    def build_pipeline(use_las_fallback):
        if use_las_fallback:
            reader = {"type": "readers.las", "filename": href}
            crop = {"type": "filters.crop", "bounds": bounds}
            return {"pipeline": [reader, crop, {"type": "writers.las", "filename": str(out_path)}]}
        return {
            "pipeline": [
                {"type": "readers.copc", "filename": href, "bounds": bounds},
                {"type": "writers.las", "filename": str(out_path)},
            ]
        }

    last_err = ""
    use_las_fallback = False
    for attempt in range(1, retries + 1):
        pipeline = build_pipeline(use_las_fallback)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pipeline, f)
            pipeline_path = f.name

        try:
            try:
                result = subprocess.run(
                    ["pdal", "pipeline", pipeline_path],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                last_err = "timed out after 300s"
                result = None
        finally:
            Path(pipeline_path).unlink()

        if result is not None and result.returncode == 0:
            return True, ""

        if result is not None:
            last_err = result.stderr
            if not use_las_fallback and "vlr" in last_err.lower():
                use_las_fallback = True
                print(f"WARNING: {href} failed with a VLR-related error under readers.copc, "
                      f"switching to readers.las+filters.crop")

        if attempt < retries:
            print(f"WARNING: attempt {attempt}/{retries} failed for {href} ({last_err.strip()[:200]}), retrying")
            time.sleep(backoff * attempt)

    return False, last_err


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
    Reproject a WGS84 bbox to Kentucky Single Zone (EPSG:3089) via a PDAL
    CLI subprocess instead of pyproj's Python bindings.

    pyproj's own log callback ('pyproj._context.pyproj_log_function') crashes
    trying to decode a non-UTF8 warning that PROJ emits for this transform,
    which corrupts the result into silently returning inf instead of raising
    -- reproducible 100% of the time on Coiled workers, and unaffected by
    forcing a UTF-8 locale, narrowing the operation search with
    area_of_interest, or disabling PROJ logging outright (all tried and
    confirmed not to help). That callback only exists because pyproj wires
    it up in Python; PDAL's own C++ PROJ integration doesn't register it, so
    it isn't exposed to the same bug even though it links the same libproj.
    Subprocess isolation also means a crash here (if it happens anyway)
    can't take down the parent process, matching crop_single_href's pattern.
    """

    minx, miny, maxx, maxy = bbox

    in_path = Path(tempfile.gettempdir()) / f"reproject_in_{os.getpid()}_{id(bbox)}.csv"
    out_path = Path(tempfile.gettempdir()) / f"reproject_out_{os.getpid()}_{id(bbox)}.csv"
    in_path.write_text(f"X,Y\n{minx},{miny}\n{maxx},{maxy}\n")

    pipeline = {
        "pipeline": [
            {"type": "readers.text", "filename": str(in_path)},
            {"type": "filters.reprojection", "in_srs": f"EPSG:{src_epsg}", "out_srs": f"EPSG:{dst_epsg}"},
            {"type": "writers.text", "filename": str(out_path), "order": "X,Y"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
        pipeline_path = f.name

    try:
        result = subprocess.run(
            ["pdal", "pipeline", pipeline_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"PDAL reprojection failed: {result.stderr.strip()[:500]}")

        out_df = pd.read_csv(out_path)
        x1, y1 = out_df.iloc[0][["X", "Y"]]
        x2, y2 = out_df.iloc[1][["X", "Y"]]
    finally:
        Path(pipeline_path).unlink(missing_ok=True)
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

    result_bbox = [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2)
    ]

    if not all(math.isfinite(v) for v in result_bbox):
        raise RuntimeError(f"reproject_bbox produced non-finite coordinates: {result_bbox} (from {bbox})")

    return result_bbox


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
    # for i in item_list:
    #     print(i)

    # Each STAC item carries other assets besides the point cloud itself
    # (e.g. a .png thumbnail) -- only the .laz is a real COPC file PDAL can
    # read, so only keep that one. Otherwise crop_single_href wastes retries
    # on files that were never valid COPC input and quarantines them for
    # nothing.
    href_list = []

    for item in item_list:
        for asset in item.assets.values():
            if asset.href.endswith(".laz"):
                href_list.append(asset.href)
    # for h in href_list:
    #     print(h)

    return href_list


def report_problematic(problematic, tile_id, bucket=S3_BUCKET):
    """
    Record which source hrefs failed to read while cropping a tile, and which
    tile they were needed for, so bad files can be tracked/reviewed later.
    One manifest per source tile, not one shared log -- many tiles may be
    processed concurrently across Coiled workers, and a single shared file
    would risk concurrent-write conflicts.
    """

    df = pd.DataFrame(problematic)
    local_path = Path(tempfile.gettempdir()) / f"{tile_id}_problematic.parquet"
    df.to_parquet(local_path)
    upload_to_s3(local_path, bucket, f"phase2/laz-problematic/{tile_id}_problematic.parquet")
    local_path.unlink()


def crop_copc(hrefs, bounds, out_laz, quarantine_bucket=S3_BUCKET):

    start = time.perf_counter()

    tile_id = Path(out_laz).stem
    tmp_dir = Path(tempfile.mkdtemp(prefix="copc_crop_"))
    good_paths = []
    problematic = []

    # Crop each href independently and in parallel (network-bound, no
    # dependency between them) so one bad file only costs its own slot,
    # not the whole tile.
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(hrefs)))) as executor:
        futures = {}
        for i, href in enumerate(hrefs):
            part_path = tmp_dir / f"part_{i}.las"
            futures[executor.submit(crop_single_href, href, bounds, part_path)] = (href, part_path)

        for future in futures:
            href, part_path = futures[future]
            ok, err = future.result()
            if ok:
                good_paths.append(part_path)
            else:
                print(f"WARNING: could not read {href}, quarantining ({err.strip()[:200]})")
                quarantined = True
                try:
                    quarantine_href(href, quarantine_bucket)
                except Exception as e:
                    quarantined = False
                    print(f"WARNING: could not quarantine {href} either: {e}")
                problematic.append({
                    "source_tile": tile_id,
                    "href": href,
                    "filename": href.rsplit("/", 1)[-1],
                    "error": err.strip()[:500],
                    "quarantined": quarantined,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    if problematic:
        report_problematic(problematic, tile_id)

    if not good_paths:
        raise RuntimeError("no readable COPC files -- nothing to crop for this tile")

    # Ensure the output directory exists -- a missing directory here throws
    # an uncaught pdal::pdal_error deep in PDAL's C++ writer, which is not
    # the same as any other Python exception: it aborts the whole process
    # (SIGABRT), killing every other task running on the same worker along
    # with this one, not just this tile. Observed in production.
    Path(out_laz).parent.mkdir(parents=True, exist_ok=True)

    # Merge the successfully-cropped parts via the PDAL CLI in a subprocess,
    # not the Python bindings -- these are local disk reads/writes so there's
    # no S3/arbiter risk, but a write failure (e.g. permissions, disk full)
    # throws the same kind of uncatchable C++ exception either way. Isolating
    # it in a subprocess means a crash here only kills that subprocess.
    #
    # writers.las, not writers.copc: multiple reader stages feeding a single
    # writers.copc are treated as separate "views", and writers.copc doesn't
    # support that -- it silently overwrites its output with each view in
    # turn, keeping only the LAST good_path's points and discarding
    # everything else (confirmed: a merge of 4 real cropped parts produced
    # an output with exactly the last file's point count). writers.las
    # concatenates multiple views correctly by default, and the octree/
    # spatial-index structure COPC provides is never needed here anyway --
    # this file is only ever read back whole, locally, by
    # calculate_point_features, not queried selectively like the original
    # S3 sources.
    merge_pipeline = {
        "pipeline": [str(p) for p in good_paths] + [{"type": "writers.las", "filename": str(out_laz)}]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(merge_pipeline, f)
        merge_pipeline_path = f.name

    try:
        result = subprocess.run(
            ["pdal", "pipeline", merge_pipeline_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        Path(merge_pipeline_path).unlink()

    if result.returncode != 0:
        raise RuntimeError(f"PDAL merge failed for {out_laz}: {result.stderr.strip()[:500]}")

    elapsed = time.perf_counter() - start
    print(f"merged {len(good_paths)}/{len(hrefs)} source files into {out_laz}")
    print(f"Elapsed: {elapsed:.1f} seconds")

    for part_path in good_paths:
        part_path.unlink()
    tmp_dir.rmdir()

    upload_to_s3(out_laz, S3_BUCKET, f"phase2/laz/{Path(out_laz).name}")


def get_buffered_tile_footprints(stac_item, url, bbox):
    """
    This function takes a bbox and returns a list of the 
    footprints of the buffered tiles that intersect with it.
    """

    output_buidlings_file = f'building-files/{stac_item.id}.parquet'
    # Worker-side write -- main.py's mkdir for this folder only runs on the
    # client machine, not on the remote Coiled worker, so it must be created
    # here too (same reasoning as crop_copc's out_laz.parent.mkdir).
    Path(output_buidlings_file).parent.mkdir(parents=True, exist_ok=True)

    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD spatial;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    query = f"""
    SELECT id, height, geometry
    FROM read_parquet('{url}', filename=true, hive_partitioning=1)
    WHERE bbox.xmin <= {xmax}
    AND bbox.xmax >= {xmin}
    AND bbox.ymin <= {ymax}
    AND bbox.ymax >= {ymin}
    """

    buildings_df = con.execute(query).df()
    print(f"wrote {len(buildings_df):,} rows to {output_buidlings_file}")

    buildings_gdf = gpd.GeoDataFrame(
    buildings_df[["id", "height"]],
    geometry=gpd.GeoSeries.from_wkb(buildings_df["geometry"].apply(bytes)),
    crs="EPSG:4326")

    print(buildings_gdf.geom_type.value_counts())

    buildings_gdf.to_parquet(output_buidlings_file)
    print(f"wrote {len(buildings_gdf):,} rows to {output_buidlings_file}")

    upload_to_s3(output_buidlings_file, S3_BUCKET, f"phase2/buildings/{stac_item.id}_buffered30m.parquet")

    return output_buidlings_file


def main():

    item_id = 'N075E299_LAS_Phase2.copc'
    out_laz = Path('lidar-files/N075E299.laz')
    # item_id = 'N075E295_LAS_Phase2.copc'
    collection = 'laz-phase2'
    stac = 'https://spved5ihrl.execute-api.us-west-2.amazonaws.com'
    item = get_stac_item(item_id, collection, stac)
    bbox = item.bbox
    # print(f'bbox {bbox}') 

    buffer = 30  # in meters
    distance = get_distance_degrees(buffer)
    # print(f'Degrees: {distance}')
    
    bbox_buffer = get_buffered_bbox(item.bbox, distance)
    print(f'Buffered bbox: {bbox_buffer}')


    hrefs = search_stac(stac, collection, bbox_buffer)
    # print(stac_search)
    # for h in hrefs:
        # print(h)

    bbox_3089 = reproject_bbox(bbox_buffer)

    print(f"3089 bbox: {bbox_3089}")

    bounds = pdal_bounds(bbox_3089)

    overture_release = '2026-07-22.0'
    overture_url = f"s3://overturemaps-us-west-2/release/{overture_release}/theme=buildings/type=building/*"

    # crop_copc and get_buffered_tile_footprints don't depend on each other's
    # output (both only need bbox_buffer/bounds), and both are dominated by
    # network wait (S3), so run them on separate threads instead of back to back.
    with ThreadPoolExecutor(max_workers=2) as executor:
        crop_future = executor.submit(crop_copc, hrefs, bounds, out_laz)
        footprints_future = executor.submit(get_buffered_tile_footprints, item, overture_url, bbox_buffer)

        crop_future.result()
        tile_buildings = footprints_future.result()

    print(f'{tile_buildings} created')


if __name__ == '__main__':
    main()
