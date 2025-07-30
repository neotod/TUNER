import numpy as np
import torch
from torch import nn
from modules.encoding import Encoding

import torch
import torch.nn.functional as F
import math
from torch.nn import init
import numpy as np
from torch import nn



class SymSumLayer(nn.Module):
    '''
    SineLayer is a custom PyTorch module that applies the Sinusoidal activation function to the output of a linear transformation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If it is the first layer, we initialize the weights differently. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.
        scale (float, optional): Scaling factor for the output of the sine activation. Default is 10.0.
        init_weights (bool, optional): If True, initializes the layer's weights according to the SIREN paper. Default is True.

    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
       
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer initialization
                bound = 1 / self.in_features
            else:
                # SIREN initialization for deeper layers
                bound = np.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
        
    def forward(self, input):
        x = self.linear(input)
        if self.is_first:
            return torch.sin(2 * self.omega_0 * x)
   
        return torch.sin(self.omega_0 * x) + input
 

    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, ffn_type=None,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30,
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None}):
        super().__init__()


        # Positional Encoding        
        if ffn_type == 'sine':
            self.pos_encode = 'gaussian'
        else:
            self.pos_encode = pos_encode_configs['type']
        
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, Gaussian]"

        
        
        self.nonlin = SymSumLayer    
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                   is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                       is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                       is_first=False, omega_0=hidden_omega_0))
        
        # self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
                    
        return output
  