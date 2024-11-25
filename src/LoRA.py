import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, TypeVar

class LoRA(nn.Module):
  """
  A low-rank adaptation layer.
  """

  def __init__(self, in_dim:int, out_dim:int, rank:int=8, lora_alpha:int=16, lora_dropout:float=0.1):
    """
    Constructor
      Args:
      -in_dim: input dimension of lora
      _out_dim: output dimension of lora
      -rank: the rank of the low-rank adapation matrix
      -lora_alpha: the scaling constant alpha of LoRA
      -lora_dropout: the dropout probability of LoRA
    """

    super(LoRA, self).__init__()
    self.rank = rank
    self.lora_alpha = lora_alpha
    self.lora_dropout = nn.Dropout(lora_dropout)
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.scaling = self.lora_alpha / self.rank

    assert self.rank > 0, "LoRA 'rank' should be an intger greater than zero!"
    assert isinstance(self.rank, int), "Variable 'rank' should be an intger greater than zero!"
    assert isinstance(self.rank, int), "LoRA 'alpha' should be an intger greater than zero!"

    self.pretrained = nn.Linear(self.in_dim, self.out_dim, bias=True)
    self.pretrained.weight.requires_grad = False

    self.lora_A = nn.Linear(self.in_dim, self.rank, bias=False)
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    self.lora_B = nn.Linear(self.rank, out_dim, bias=False)
    nn.init.constant_(self.lora_B.weight, 0)

  def forward(self, x):
    pretrained_out = self.pretrained(x)
    lora_out = self.lora_A(x)
    lora_out = self.lora_B(lora_out)
    lora_out = lora_out * self.scaling
    lora_out = self.lora_dropout(lora_out)
    return pretrained_out + lora_out


class lora_adapter(nn.Module):
  """
  A lora adapter layer .
  """
  def __init__(self, nn_module, rank:int=8, lora_alpha:int=16, lora_dropout:float=0.1):
    """
    Constructor
      Args:
      -nn_module: a torch.nn module which the lora adapter is to replace
      -rank: the rank of the low-rank adapation matrix
      -lora_alpha: the scaling constant alpha of LoRA
      -lora_dropout: the dropout probability of LoRA
    """
    super(lora_adapter, self).__init__()

    out_dim, in_dim = nn_module.weight.shape

    self.lora = LoRA(in_dim, out_dim, rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    with torch.no_grad():
      self.lora.pretrained.weight.copy_(nn_module.weight)
      self.lora.pretrained.bias.copy_(nn_module.bias)

  def forward(self, x):
    lora_out = self.lora(x)
    return lora_out


def add_lora(model, module_type, module_names, rank: int=8, lora_alpha: int=16, lora_dropout: float=0.1,):
  for name, module in model.named_children():
    if isinstance(module, module_type) and name in module_names:
      print(name, module)
      lora = lora_adapter(module, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
      setattr(model, name, lora)
    else:
      add_lora(module, module_type, module_names, rank=8, lora_alpha=16, lora_dropout=0.1)