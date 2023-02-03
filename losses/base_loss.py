import torch
import torch.nn as nn
from registry import LossRegistry

class BaseLoss(nn.Module, LossRegistry, loss_name='ignore'):
    pass