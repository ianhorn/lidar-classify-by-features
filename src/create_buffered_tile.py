#!/usr/bin/env python3

T"""
This script uses the bounding box of kyfromabove lidar tile, buffers it by a 
specfied distance, uses the bufferred bounding box to search stac for lidar tiles.
The points from each lidar will be read to create a new temporary local tile. 
"""

import json
import requests
from pystac import Item
from pathlib import Path
from pystac_client import ItemSearch


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
        maxx - distance_degrees,
        maxy - distance_degrees
    ]
    return buffered_bbox


def search_stac(stac_api, collection, buffered_bbox):

    url = f'{stac_api}/search'

    payload = {
        "collections": [collection],
        "bbox": buffered_bbox
    }

    headers = {
        "accept": "application/geo+json",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    print(f'Response Status: {response.status_code}')
    # print(json.dumps(data, indent=2))


def main():

    item_id = 'N075E299_LAS_Phase2.copc'
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

    print(f'Item: {item}\n')

    stac_search = search_stac(stac, collection, bbox_buffer)

if __name__ == '__main__':
    main()
