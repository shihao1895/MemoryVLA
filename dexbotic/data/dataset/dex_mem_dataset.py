from dexbotic.data.dataset.dex_dataset import DexDataset


class DexMemDataset(DexDataset):
    default_keys = ['input_ids', 'labels', 'action', 'image']

    def __init__(self,
                 data_args,
                 **kwargs):
        """Args:
            data_args: argparse.Namespace, the arguments for the dataset
        """
        super().__init__(data_args, **kwargs)

    def __getitem__(self, idx) -> dict:
        try:
            sample = self.unsafe_getitem(idx)
            sample["indexes"] = self.global_index[idx]
            return sample
        except Exception as e:
            # for memory, return random is forbidden
            raise RuntimeError(f"Error in getting item {idx}: {e}")
