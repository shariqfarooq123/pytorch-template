import torch
import torch.nn as nn
from utils.model_io import load_state_from_resource
from registry import ModelRegistry

class BaseModel(nn.Module, ModelRegistry, name='ignore'):
    def __init__(self):
        super().__init__()

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = [{'params': self.parameters(), 'lr': lr}]
        return param_conf

