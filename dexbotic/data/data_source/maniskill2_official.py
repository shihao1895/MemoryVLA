import math

from dexbotic.data.data_source.register import register_dataset


MANISKILL2_DATASET = {
    "pickcube": {
        "data_path_prefix": "./data/maniskill/video/PickCube-v0/base/",
        "annotations": "./data/maniskill/jsonl/PickCube-v0",
        "frequency": 1,
    },
    "stackcube": {
        "data_path_prefix": "./data/maniskill/video/StackCube-v0/base/",
        "annotations": "./data/maniskill/jsonl/StackCube-v0",
        "frequency": 1,
    },
    "picksingleycb": {
        "data_path_prefix": "./data/maniskill/video/PickSingleYCB-v0/base/",
        "annotations": "./data/maniskill/jsonl/PickSingleYCB-v0",
        "frequency": 1,
    },
    "picksingleegad": {
        "data_path_prefix": "./data/maniskill/video/PickSingleEGAD-v0/base/",
        "annotations": "./data/maniskill/jsonl/PickSingleEGAD-v0",
        "frequency": 1,
    },
    "pickclutterycb": {
        "data_path_prefix": "./data/maniskill/video/PickClutterYCB-v0/base/",
        "annotations": "./data/maniskill/jsonl/PickClutterYCB-v0",
        "frequency": 1,
    },
}

meta_data = {
    'non_delta_mask': [6],
    'periodic_mask': [3, 4, 5],
    'periodic_range': 2 * math.pi,
}

register_dataset(MANISKILL2_DATASET, meta_data=meta_data, prefix='maniskill')