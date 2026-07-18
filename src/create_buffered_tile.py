#!/usr/bin/env python3

T"""
This script uses the bounding box of kyfromabove lidar tile, buffers it by a 
specfied distance, uses the bufferred bounding box to search stac for lidar tiles.
The points from each lidar will be read to create a new temporary local tile. 
"""


from pystac import Item
from pathlib import Path
from shapely.geometry import shape
from pystac_client import ItemSearch


def get_distance_degrees(meters: float):
    meters_per_degree = 111_320
    distance_meters = meters
    distance_degrees = distance_meters / meters_per_degree

    return distance_degrees

def get_buffered_geometry(item_id: str, item_collection: str, item_api_url: str, distance_degrees: float):
    """
    requires the STAC API url, stac title
    """

    href = f'{item_api_url}/collections/{item_collection}/items/{item_id}'

    item = Item.from_file(href)  # pull the item with pystac
    geometry = item.geometry    # get the geometry
    print(f'Geometry: {geometry}\n')                  # print for confirmation

    # buffer the bbox by a distance
    buffered_geometry = shape(item.geometry).buffer(
        distance_degrees,
        cap_style="square"
    )
    print(f'Buffer: {buffered_geometry}\n')                   # print for confirmation

    return buffered_geometry


def search_stac(buffered_geometry):

    geometry = get_buffered_geometry(item, collection, stac, distance)
#     stac_string = f'{stac}/'

    return geometry


def main():

    item = 'N075E299_LAS_Phase2.copc'
    collection = 'laz-phase2'
    stac = 'https://spved5ihrl.execute-api.us-west-2.amazonaws.com'

    buffer = 30
    distance = get_distance_degrees(buffer)
    buffered_geometry = get_buffered_geometry(item, collection, stac, distance)

    print(f'Item: {item}\n')

if __name__ == '__main__':
    main()
