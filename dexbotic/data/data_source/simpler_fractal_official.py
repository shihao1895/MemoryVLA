from dexbotic.data.data_source.register import register_dataset

SIMPLER_FRACTAL_DATASET = {
    "all": {
        "data_path_prefix": "./data/fractal/video",
        "annotations": './data/fractal',
        "frequency": 1,
    },

}

meta_data = {
    'non_delta_mask': [6],
    'periodic_mask': None,
    'periodic_range': None
}


register_dataset(SIMPLER_FRACTAL_DATASET, meta_data=meta_data, prefix='simpler_fractal')
