#!/usr/bin/env python

import numpy as np
import torch
from torch import nn
import math

class Parasin(nn.Module):
    """
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
    discussion of omega_0.

    If is_first=True, omega_0 is a frequency factor which simply multiplies
    the activations before the nonlinearity. Different signals may require
    different omega_0 in the first layer - this is a hyperparameter.

    If is_first=False, then the weights will be divided by omega_0 so as to
    keep the magnitude of activations constant, but boost gradients to the
    weight matrix (see supplement Sec. 1.5)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        scale=10.0,
        init_weights=True,
    ):
        super().__init__()

        self.nf = 5
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize weights and parameters
        self.init_params()
                
    def init_params(self):
        """
        Initializes the parameters for the sinusoidal activations.
        """
        ws = self.omega_0 * torch.rand(self.nf)
        self.ws = nn.Parameter(ws, requires_grad=True)

        # Initialize phases uniformly in the range [-π, π]
        self.phis = nn.Parameter(
            -math.pi + 2 * math.pi * torch.rand(self.nf), requires_grad=True
        )

        # Initialize scale factors based on a Laplace distribution
        diversity_y = 1 / (2 * self.nf)
        laplace_samples = torch.distributions.Laplace(0, diversity_y).sample((self.nf,))
        self.bs = nn.Parameter(torch.sign(laplace_samples) * torch.sqrt(torch.abs(laplace_samples)), requires_grad=True)

    def forward(self, input):
        
        linout = self.linear(input)
        y = self.bs * torch.sin(self.ws * linout.unsqueeze(-1) + self.phis)
        
        return y.sum(dim=-1) if self.is_first else y.sum(dim=-1) + input

        
        

class INR(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = Parasin

        self.net = []
        self.net.append(
            self.nonlin(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                scale=scale,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    scale=scale,
                )
            )

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    scale=scale,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)

        return output
