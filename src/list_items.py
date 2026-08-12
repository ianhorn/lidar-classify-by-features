#!/usr/bin/env python

############################################################
# List every STAC item in a collection (e.g. laz-phase2) and
# export id/bbox/href/datetime to a Parquet manifest -- one
# row per tile, for looping over tiles in a batch/Coiled run.
# Ian Horn
# August 11, 2026
############################################################

from pathlib import Path

import pandas as pd
from pystac_client import Client

from create_buffered_tile import upload_to_s3, S3_BUCKET


def list_collection_items(stac_api: str, collection: str):
    """
    List every item in a STAC collection (no bbox filter) and return
    one row per item: id, bbox, datetime, and the pointcloud asset href.
    """

    client = Client.open(f'{stac_api}/')
    # STAC API pagination here is cursor-based (each page's "next" link embeds
    # the last item id), so pages can't be fetched concurrently -- but the
    # default page size is only 10 items/request. Bumping it to 1000 (server
    # accepts up to ~2000, rejects 10000 as "Request Entity Too Large") cuts
    # the number of sequential round-trips by ~100x.
    search = client.search(collections=[collection], limit=1000)

    rows = []
    for item in search.items():
        minx, miny, maxx, maxy = item.bbox
        rows.append({
            "id": item.id,
            "collection": collection,
            "datetime": item.datetime,
            "href": item.assets["pointcloud"].href,
            "bbox_minx": minx,
            "bbox_miny": miny,
            "bbox_maxx": maxx,
            "bbox_maxy": maxy,
        })

    print(f"{len(rows):,} items found in '{collection}'")
    return pd.DataFrame(rows)


def main():

    stac = 'https://spved5ihrl.execute-api.us-west-2.amazonaws.com'
    phase = 'laz-phase2'
    repo_root = Path(__file__).resolve().parent.parent
    output_file = repo_root / "stac_item_list.parquet"

    items_df = list_collection_items(stac, phase)

    items_df.to_parquet(output_file, compression="zstd")
    print(f"wrote {len(items_df):,} rows to {output_file}")

    upload_to_s3(output_file, S3_BUCKET, f"phase2/{output_file.name}")


if __name__ == '__main__':
    main()
