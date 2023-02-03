
# MIT License

# Copyright (c) 2022 Shariq Farooq Bhat

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TVT
from registry import DatasetRegistry

class BaseDataset(Dataset, DatasetRegistry, name='ignore'):
    def __init__(self, config, **kwargs):
        """
        Base class for all datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """
        self.config = {**config, **kwargs}
        self.kwargs = kwargs

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    def build_loader(cls, config, **kwargs):
        """Returns a data loader for the specified dataset

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): modes for dataset
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.

        Returns:
            torch.utils.data.DataLoader: Data loader for the specified dataset
        """

        samples = cls(config, **kwargs)

        if config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                samples)
        else:
            train_sampler = None

        dataloader = DataLoader(samples,
                            batch_size=config.batch_size,
                            shuffle=(train_sampler is None) and config.get('shuffle', True),
                            num_workers=config.get('workers', 0),
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            sampler=train_sampler)
        return dataloader
