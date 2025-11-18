import os
import pandas as pd
import json
from exp_helper import parse_args_from_config, load_config

#import yaml config and parse dataset root
cfg = load_config('defaults.yaml')
cfg = parse_args_from_config(cfg)

dataRoot = cfg['exp_details']['data_root']

#open whitelist json file
with open("utils\\nacc_column_whitelist.json") as file:
    keep_list = json.load(file)["columns"]

#read in NACC csv & drop columns
inputPath = os.path.join(dataRoot, "nacc_data_updated.csv")
dataframe = pd.read_csv(inputPath)

dataframe = dataframe[keep_list]

#Create new csv
outputPath = os.path.join(dataRoot, "nacc_data_dropped.csv")
dataframe.to_csv(outputPath, index=False)

print("Successfully dropped columns from NACC CSV file")

