import pandas as pd
import torch
from modelV2.tune import HyperParamOptimizer
from modelV2.train import tuning_wrapper
# from modelV2.arch.rn_2 import ResNet18
from torchvision.transforms import v2
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def extract_masked_path(target_mask_factor: str, mask_factors_str: str, masked_paths_str: str):
    try:
        # parse the mask factors/masked path strings
        mask_factors_list = parse_mask_factors(mask_factors_str)
        mask_paths_list = parse_masked_paths(masked_paths_str)

        # find the index of the target mask factor
        target_idx = mask_factors_list.index(target_mask_factor)

        # extract the target masked path
        return mask_paths_list[target_idx]
    
    except Exception as e:
        print(e)
        return 'CAUGHT_ERROR'

def parse_mask_factors(mask_factors_str: str):
    # remove brackets and spaces from the string
    mask_factors_str = mask_factors_str.translate({ord(c): None for c in "][ "})
    # the split on commas
    return mask_factors_str.split(',')

def parse_masked_paths(masked_paths_str: str):
    # split on dollar characters (pls dont have any of these in your paths!!)
    return masked_paths_str.split('$')


SEED = 49713
BASE_DIR = "/mnt/NAS3/DataBalance/balance-study/data/images/run_e1_a_1200_800/"
MASK_STR = "0_0"
BALANCE_DISTRIBUTIONS = False

N_BOOTSTRAPS = 5
EPOCHS_PER_RUN = 20
MONITOR_METRIC = 'loss'
HPARAM_RANGE_DICT = {
    'learning_rate': (0.00001, 0.01),
}


BATCH_SIZE = 48
N_RANDOM = 5
N_GUIDED = 10
MODEL_NAME = f"rn18_r3_a"

pos_df = pd.read_csv(os.path.join(BASE_DIR, "pos_out_df.csv"))
neg_df = pd.read_csv(os.path.join(BASE_DIR, "neg_proc_df.csv"))

# only include exams we've assigned to the tuning set
# this is done so evaluation class balance is consistent 
# between train/val/test sets
# 10/90 balance testing can be done after tuning is completed
neg_df = neg_df[neg_df["include_in_tuning"] == 1]

pos_df.dropna(subset=['masked_factors', 'masked_png_paths'], inplace=True)
neg_df.dropna(subset=['masked_factors', 'masked_png_paths'], inplace=True)
print(pos_df[['assigned_split']].value_counts())
print(neg_df[['assigned_split']].value_counts())

def correct_paths_to_kraken(hiti_path):
    replace_str = r"/mnt/NAS3/DataBalance/balance-study\1"
    # use re.sub to replace everything before '/data/images'
    return re.sub(r'^.*?(/data/images)', replace_str, hiti_path)

pos_df['__target_path'] = pos_df.apply(
    lambda x: correct_paths_to_kraken(extract_masked_path(
        MASK_STR.replace('_', '.'), 
        x.masked_factors, 
        x.masked_png_paths
    )), 
    axis=1
)

neg_df['__target_path'] = neg_df.apply(
    lambda x: correct_paths_to_kraken(extract_masked_path(
        MASK_STR.replace('_', '.'), 
        x.masked_factors, 
        x.masked_png_paths
    )), 
    axis=1
)

DF_POOL_DICT = {
    "pos": {
        "train": pos_df[pos_df.assigned_split == "train"],
        "val": pos_df[pos_df.assigned_split == "val"],
        "test": pos_df[pos_df.assigned_split == "test"]
    },
    "neg": {
        "train": neg_df[neg_df.assigned_split == "train"],
        "val": neg_df[neg_df.assigned_split == "val"],
        "test": neg_df[neg_df.assigned_split == "test"]
    }
}