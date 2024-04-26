import torch
from modelV2.tune import HyperParamOptimizer
from modelV2.train import tuning_wrapper
from modelV2.arch.rn_2 import ResNet18
from torchvision.transforms import v2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SEED = 12763
EPOCHS_PER_RUN = 20
MONITOR_METRIC = 'loss'
HPARAM_RANGE_DICT = {
    'dropout_proportion': (0.1, 0.9), 
    'learning_rate': (0.00001, 0.01),
}

MASK_STR = "0_0"

BASE_PATH = "/data/beatrice/masked_roi/mammo_masked_roi/data/images/run_e1_a_1200_800"
MASK_FACTOR = f"masked_{MASK_STR}x"
BATCH_SIZE = 48
N_RANDOM = 5
N_GUIDED = 10
MODEL_NAME = f"masked_{MASK_STR}_rn18_r3_a"
SET_DIR_DICT = {
    "train": "train", 
    "val": "val", 
    "test": "test"
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
    monitor_metric=MONITOR_METRIC, 
    epochs_per_run=EPOCHS_PER_RUN,
    seed=SEED
)

# set device and model
hparam_optimizer.set_model(
    device=device, 
    model_func=ResNet18
)

# load data into optimizer
hparam_optimizer.load_data(
    base_path=BASE_PATH, 
    mask_factor=MASK_FACTOR, 
    set_dir_dict=SET_DIR_DICT, 
    batch_size=BATCH_SIZE,
    augment_dict=AUGMENT_DICT
)

# start optimizer
hparam_optimizer.optimize(
    objective=tuning_wrapper,
    n_random=N_RANDOM, 
    n_guided=N_GUIDED, 
    model_name=MODEL_NAME
)
