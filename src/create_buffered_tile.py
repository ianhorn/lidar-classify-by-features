#!/usr/bin/env python3

T"""
This script uses the bounding box of kyfromabove lidar tile, buffers it by a 
specfied distance, uses the bufferred bounding box to search stac for lidar tiles.
The points from each lidar will be read to create a new temporary local tile. 
"""


import json
import pdal
import time
import requests
from pystac import Item
from pathlib import Path
from pyproj import Transformer
from pystac_client import Client


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


def crop_copc(hrefs, bounds, out_laz):

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
