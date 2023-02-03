from datasets.base_dataset import BaseDataset 
import numpy as np

class ToyDataset(BaseDataset, name='toy_dataset'):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = {**config, **kwargs}
        self.kwargs = kwargs

        self.numbers = np.arange(100)
    
    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        return self.numbers[idx]