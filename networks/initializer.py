from itertools import product
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from mrnet.networks.mrnet import MRFactory
import torch.distributions as dist
from utils import create_clamps


class Initializer:
    def __init__(self, hyper, init_freqs=[None, None], bias_init=True,
                 init_W=True, sample_path=None):
        self.hyper = hyper

        if hyper['period']:
            self.factor = 2 * torch.pi / hyper['period']
            self.initialize_first_layer(init_freqs, sample_path=sample_path)
        else:
            self.factor = torch.pi / hyper['omega_0']
            self.initialize_first_layer(init_freqs, sample_path=sample_path)
            # self.model = MRFactory.from_dict(self.hyper)
        if bias_init:
            self.initialize_bias()
        if self.hyper['bounds'][0] and init_W:
            self.initialize_middle_layer()
        self._input_frequencies()

    def initialize_first_layer(self, init_freqs_1d=[None, None],
                               sample_path=None):
        low_freqs_1d, high_freqs_1d = init_freqs_1d
        if init_freqs_1d[0] is None:
            low_freqs_1d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if init_freqs_1d[1] is None:
            high_freqs_1d = [0, 32, 64, 96, 128, 160, 192, 224, 256]
        self.hyper['init_freqs'] = [*low_freqs_1d, *high_freqs_1d]
        possible_frequencies = []
        for init_freqs_1d in [low_freqs_1d, high_freqs_1d]:
            freqs_1d = init_freqs_1d + [-i for i in init_freqs_1d]
            possible_frequencies.append(torch.tensor(list(product(freqs_1d,
                                                                  repeat=2))))
        possible_frequencies = torch.cat(possible_frequencies)

        chosen_frequencies = []
        for x, y in possible_frequencies:
            if x < 0 or (x == 0 and y <= 0):
                continue
            chosen_frequencies.append([x, y])
        chosen_frequencies = torch.tensor(
            chosen_frequencies
            ).reshape(-1, self.hyper['in_features'])

        # self.hyper['hidden_features'][0][0] = chosen_frequencies.shape[0] + 2
        # print(f'\nHidden matrix size: {self.hyper["hidden_features"][0][1]}x' +
            #   f'{self.hyper["hidden_features"][0][0]}\n')
        self.model = MRFactory.from_dict(self.hyper)

        # self.model.stages[0].first_layer.linear.weight = \
        #     torch.nn.Parameter(
        #     chosen_frequencies.float()[:self.hyper['hidden_features'][0][0] - 2],
        #     requires_grad=False
        # )
        nf = self.factor * torch.cat([
            chosen_frequencies.float()[:self.hyper['hidden_features'][0][0] - 2],
            torch.eye(self.hyper['in_features'])
            ], axis=0).reshape(-1, 2)
        self.model.stages[0].first_layer.linear.weight = torch.nn.Parameter(
            nf,
            requires_grad=False
        )

    def _input_frequencies(self):
        self.model.frequencies = (
            1 / self.factor *
            self.model.stages[0].first_layer.linear.weight
        )

    def initialize_bias(self):
        for stage in range(self.model.n_stages()):
            stage_first_layer = self.model.stages[stage].first_layer.linear
            with torch.no_grad():
                torch.nn.init.uniform_(
                    stage_first_layer.bias,
                    -torch.pi/2,
                    torch.pi/2
                )

    def initialize_middle_layer(self):
        omega = self.model.stages[0].first_layer.linear.weight
        W = self.model.stages[-1].middle_layers[0].linear.weight

        chosen_frequencies = self.model.period / (2 * torch.pi) * omega
        device = 'cuda:0' if self.hyper['device'] == 'cuda' else 'cpu'
        clamps = create_clamps(self.hyper['bounds'][0],
        # clamps = create_clamps([1],
                               self.hyper['block_limits'],
                               chosen_frequencies,
                               device)

        new_W = dist.Normal(torch.zeros_like(clamps, device=device),
                            0.3 * clamps / self.hyper['hidden_omega_0']
                            ).sample([W.shape[0]])
        # new_W = clamps / hyper['hidden_omega_0'] * (2 * torch.rand_like(W,
        # device=device) - 1)
        self.model.stages[-1].middle_layers[0].linear.weight = \
            torch.nn.Parameter(new_W)

    def get_model(self):
        return self.model

    def initialize_middle_layer_with_identity(self, epsilon=1e-4):
        omega = self.model.stages[0].omega_G
        for i in range(0, len(self.model.stages[0].middle_layers)):
            W = torch.zeros_like(
                self.model.stages[0].middle_layers[i].linear.weight,
                requires_grad=False)
            min_dim = min(W.shape)
            W[:min_dim, :min_dim] = epsilon * torch.eye(min_dim) / (omega)
            with torch.no_grad():
                self.model.stages[0].middle_layers[i].linear.weight = \
                    torch.nn.Parameter(W, requires_grad=True)
