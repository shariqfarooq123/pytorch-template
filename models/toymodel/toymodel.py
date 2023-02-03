import torch
import torch.nn as nn
from models.base_model import BaseModel

class ToyModel(BaseModel, name='toymodel'):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs