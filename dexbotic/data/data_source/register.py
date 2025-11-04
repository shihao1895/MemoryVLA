CONVERSATION_DATA = {}


def register_dataset(dataset, prefix='', meta_data=None):
    if prefix:
        dataset = {f'{prefix}_{k}': v for k, v in dataset.items()}
    if meta_data is not None:
        new_dataset = {}
        for k, v in dataset.items():
            new_dataset[k] = v
            if 'meta_data' not in v:
                new_dataset[k]['meta_data'] = meta_data
        dataset = new_dataset
    CONVERSATION_DATA.update(dataset)
