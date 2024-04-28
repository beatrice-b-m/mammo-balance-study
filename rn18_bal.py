import pandas as pd
pd.options.mode.chained_assignment = None
import torch
from modelV2.tune import HyperParamOptimizer
from modelV2.train import tuning_wrapper
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


SEED = 87231
BASE_DIR = "/mnt/NAS3/DataBalance/balance-study/data/images/run_e1_a_1200_800/"
MASK_STR = "0_0"

N_BOOTSTRAPS = 5
EPOCHS_PER_RUN = 20
MONITOR_METRIC = 'loss'
HPARAM_RANGE_DICT = {
    'learning_rate': (0.00001, 0.01),
}
BATCH_SIZE = 24
N_RANDOM = 5
N_GUIDED = 10

# assign model type and get its name
MODEL_TYPE = 'resnet18'
TEST_NUM = 'a'
BALANCE_DISTRIBUTIONS = True

if BALANCE_DISTRIBUTIONS:
    name_template = "{}_bal_{}"
else:
    name_template = "{}_no_bal_{}"
    
model_name = name_template.format(MODEL_TYPE, TEST_NUM)


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

AUGMENT_DICT = {
    'crop': {
        'enabled': True, 
        'func': v2.RandomResizedCrop,
        'prob': 1.0,
        'params': {
            'size': (1200, 800), 
            'scale': (0.7, 1.0),
        }
    },
    'rotation': { 
        'enabled': True, 
        'func': v2.RandomRotation,
        'prob': 0.4,
        'params': {
            'degrees': 5
        }
    },
    'color_jitter': {
        'enabled': True,
        'func': v2.ColorJitter,
        'prob': 0.2,
        'params': {
            'brightness': 0.4, 
            'contrast': 0.4,
            'saturation': 0.4, 
            'hue': 0.2
        }
    },
    'gaussian_blur': {
        'enabled': True,
        'func': v2.GaussianBlur,
        'prob': 0.2,
        'params': {
            'kernel_size': 3
        }
    }
}

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device...")

# get optimizer
hparam_optimizer = HyperParamOptimizer(
    hparam_range_dict=HPARAM_RANGE_DICT, 
    balance=BALANCE_DISTRIBUTIONS,
    monitor_metric=MONITOR_METRIC,
    n_bootstraps=N_BOOTSTRAPS,
    epochs_per_run=EPOCHS_PER_RUN,
    seed=SEED
)

# set device and model
hparam_optimizer.set_model(
    device=device, 
    model_type='resnet18'
)

# load data into optimizer
hparam_optimizer.load_data(
    df_pool_dict=DF_POOL_DICT,
    batch_size=BATCH_SIZE,
    augment_dict=AUGMENT_DICT
)

# start optimizer
hparam_optimizer.optimize(
    objective=tuning_wrapper,
    n_random=N_RANDOM, 
    n_guided=N_GUIDED, 
    model_name=model_name
)