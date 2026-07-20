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


def search_stac(stac_api, collection, buffered_bbox):
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
        print(i.id)

    href_list = []

    for item in item_list:
        for asset in item.assets.values():
            href_list.append(asset.href)
    for h in href_list:
        print(h)

    return href_list

# def process_copc(copc):

#     json = {

#     }



def main():

    item_id = 'N075E299_LAS_Phase2.copc'
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

    stac_search = search_stac(stac, collection, bbox_buffer)


if __name__ == '__main__':
    main()
