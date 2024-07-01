import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from networks.siren import SineLayer


class Pruner:
    def __init__(self, model, p, dim=1):
        self.model = model
        self.dim = dim
        self.p = p


    def prune(self, percentages_to_prune):
        for i, (_, module) in enumerate(self.model.named_modules()):
            amount = percentages_to_prune[i]
            if isinstance(module, SineLayer):
                prune.ln_structured(module, name='weight', amount=amount,
                                    dim=self.dim, p=self.p)