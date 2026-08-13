#!/usr/bin/env python3

import coiled
import pandas as pd
import create_buffered_tile as cbt
import calculate_point_features as cpf

df = pd.read_parquet('s3://lidar-classification/phase2/stac_item_list.parquet')

print(df)


