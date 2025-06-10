from typing import Sequence
from warnings import warn
import torch
import numpy as np

from torch import nn
from collections import OrderedDict

RANDOM_SEED = 777


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def n_cartesian_freqs(high_list, in_feats=2):
    # half of cartesian product of list [*high_list, *-high_list]
    # where high_list is a list of positive integers.
    if 0 in high_list:
        return ((2 * len(high_list)) ** in_feats) / 2
    else:
        return ((2 * len(high_list) + 1) ** in_feats) / 2


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
    # discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies
    # the activations before the nonlinearity. Different signals may require
    # different omega_0 in the first layer - this is a hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to
    # keep the magnitude of activations constant, but boost gradients to the
    # weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 omega_0=30, period=0, mode="sampling", **kwargs):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mode = mode

        self.__set_bandlimit(kwargs.get('bandlimit', 0))
        self.__set_low_range(kwargs.get('low_range', 0))
        self.__set_perc_low_freqs(kwargs.get('perc_low_freqs', 0.7))

        self.period = period

        if 'bounds' in kwargs and kwargs['bounds']:
            self.class_bounds = kwargs['bounds']  # UNDO
            self.bound= nn.Parameter(
                0.5 * torch.ones((self.in_features), requires_grad=True))

        self.init_weights()

    def init_weights(self):
        if self.period > 0:
            if self.mode == "uniform":
                self.init_periodic_uniform()
            else:
                self.init_periodic_sampling()

        else:
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1, 1)  # *self.omega_0
                else:
                    self.linear.weight.uniform_(
                        -np.sqrt(6 / self.in_features) / self.omega_0,
                        np.sqrt(6 / self.in_features) / self.omega_0)

    def init_periodic_uniform(self, used_weights=[]):
        # don't need to choose the origin
        if self.in_features == 2:
            discarded_freqs = set([(0, 0)])
        else:
            discarded_freqs = set()
        discarded_freqs = discarded_freqs.union(set(used_weights))

        with torch.no_grad():
            if self.is_first:
                rng = np.random.default_rng(RANDOM_SEED)
                if self.in_features == 2:
                    possible_frequencies = cartesian_product(
                        np.arange(0, self.omega_0 + 1),
                        np.arange(-self.omega_0, self.omega_0 + 1))
                else:
                    possible_frequencies = cartesian_product(
                        *(self.in_features * [
                            np.array(range(-self.omega_0,
                                           self.omega_0 + 1))])
                    )
                if discarded_freqs:
                    possible_frequencies = np.array(list(
                        set(tuple(map(
                            tuple, possible_frequencies
                            ))) - set(used_weights)
                    ))
                chosen_frequencies = torch.from_numpy(
                    rng.choice(possible_frequencies, self.out_features, False)
                )

                self.linear.weight = nn.Parameter(
                    chosen_frequencies.float() * 2 * torch.pi / self.period)
                # first layer will not be updated during training
                self.linear.weight.requires_grad = False

    def init_periodic_sampling(self, used_weights=[]):
        # don't need to choose the oigin
        if self.in_features == 2:
            discarded_freqs = set([(0, 0)])
        else:
            discarded_freqs = set()
        discarded_freqs = discarded_freqs.union(set(used_weights))

        with torch.no_grad():
            if self.is_first:
                rng = np.random.default_rng(RANDOM_SEED)

                n_freqs = self.out_features - self.in_features
                n_low_freqs = round(self.__perc_low_freqs * n_freqs)
                n_high_freqs = n_freqs - n_low_freqs
                in_feats, blimit = self.in_features, self.__bandlimit
                l_blimit = self.__low_range
                assert n_freqs > 0

                low_freqs_1d = np.arange(-l_blimit, l_blimit + 1)
                possible_low_freqs = cartesian_product(
                    low_freqs_1d[l_blimit:],
                    *np.tile(low_freqs_1d, (in_feats - 1, 1))
                    )
                
                # n_high_freqs_1d := N° of high freqs along x axis in quadrant I 
                n_high_freqs_1d =int(
                    (n_high_freqs / (2 ** (in_feats - 1))) ** (1 / in_feats)
                    ) + 1
                high_freqs_1d = np.arange(
                    -blimit, blimit + 1, max(int(blimit / max(n_high_freqs_1d, 1)), 1)
                    )
                possible_high_freqs = cartesian_product(
                    high_freqs_1d[:n_high_freqs_1d + 2],
                    *np.tile(high_freqs_1d, (in_feats - 1, 1))
                    )

                chosen_low_freqs = torch.from_numpy(
                    rng.choice(possible_low_freqs, n_low_freqs, True)
                    )
                try:
                    chosen_high_freqs = torch.from_numpy(
                        rng.choice(possible_high_freqs, n_high_freqs, False)
                        )
                except ValueError:
                    chosen_high_freqs = torch.from_numpy(
                        rng.choice(possible_high_freqs, n_high_freqs, True)
                        )
                chosen_frequencies = torch.cat(
                    (torch.eye(in_feats), chosen_low_freqs, chosen_high_freqs)
                    )
                self.linear.weight = nn.Parameter(
                    chosen_frequencies.float() * 2 * torch.pi / self.period
                    )
                # first layer will not be updated during training
                self.linear.weight.requires_grad = False

                if hasattr(self, 'bounds'):
                    bounds = torch.cat([
                        self.class_bounds[0] * torch.ones(n_low_freqs +
                                                          self.in_features),
                        self.class_bounds[1] * torch.ones(n_high_freqs)
                    ])
                    self.bound = nn.Parameter(bounds)

    def forward(self, input):
        if self.period > 0 and self.is_first:
            x = self.linear(input)
        else:
            x = self.omega_0 * self.linear(input)
        return torch.sin(x)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

    def list_1d_high_freqs(self):
        high_list = [0]
        perc_high_freqs = n_cartesian_freqs(high_list, self.in_features) / self.out_features
        while perc_high_freqs < 1 - self.__perc_low_freqs:
            high_list.append(0)
            perc_high_freqs = n_cartesian_freqs(high_list, self.in_features) / self.out_features
        number_high_freqs = len(high_list)
        step = round((self.__bandlimit) / number_high_freqs)
        step = max(step, 1)
        high_list = [0] + [
            self.__low_range + i * step for i in range(number_high_freqs)
            ]
        high_list[-1] = self.__bandlimit
        return high_list

    def __set_bandlimit(self, var):
        b = int(var)
        if b > 0:
            self.__bandlimit = b
        else:
            self.__bandlimit = int(self.omega_0 / 3)

    def __set_low_range(self, var):
        low_limit = int(var)
        if low_limit > 0 and low_limit < self.__bandlimit:
            self.__low_range = var
        else:
            self.__low_range = 12

    def __set_perc_low_freqs(self, var):
        if var > 0 and var <= 1:
            self.__perc_low_freqs = var
        else:
            self.__perc_low_freqs = 0.7
            print(f"Percentage of low frequencies {var} must be in [0, 1]." +
                  f"Setting bandlimit to {self.__perc_low_freqs}.")


class Siren(nn.Module):
    """
    This SIREN version comes from:
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, first_omega_0, hidden_omega_0,
                 bias=True, outermost_linear=True, superposition_w0=True, 
                 **kwargs):
        super().__init__()

        if not isinstance(hidden_features, Sequence):
            hidden_features = [hidden_features] * (hidden_layers + 1)

        hidden_idx = 0
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[hidden_idx], bias=bias,
                                  is_first=True, omega_0=first_omega_0,
                                  **kwargs))

        self.n_layers = hidden_layers + 1
        while hidden_idx < hidden_layers:
            self.net.append(SineLayer(hidden_features[hidden_idx],
                                      hidden_features[hidden_idx + 1],
                                      is_first=False, omega_0=hidden_omega_0))
            hidden_idx += 1

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[hidden_idx], out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features[hidden_idx]) / hidden_omega_0,
                    np.sqrt(6 / hidden_features[hidden_idx]) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[hidden_idx], out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def reset_weights(self):
        def reset_sinelayer(m):
            if isinstance(m, SineLayer):
                m.init_weights()
        self.apply(reset_sinelayer)

    def forward(self, coords):
        # allows to take derivative w.r.t. input
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__),
                                      "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__),
                                  "%d" % activation_count))] = x
            activation_count += 1

        return activations


class ReLULayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: int = 30, period: float = 0,
                 **kwargs):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.period = period

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        if self.period > 0 and self.is_first:
            x = self.linear(input)
        else:
            x = self.omega_0 * self.linear(input)
        return F.relu(x)


class TanhLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: int = 30, period: float = 0,
                 **kwargs):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.period = period

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        if self.period > 0 and self.is_first:
            x = self.linear(input)
        else:
            x = self.omega_0 * self.linear(input)
        return F.tanh(x)


class FFMLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: int = 30, period: float = 0,
                 **kwargs):
        super().__init__()
        self.omega_0 = omega_0
        self.period = period

        self.in_features = in_features
        self.out_features = out_features
        n_freqs = self.out_features // 2

        std = np.sqrt(self.omega_0)
        self.freqs = nn.init.normal_(torch.empty(n_freqs, in_features),
                                mean=0, std=std)
        # new_linear = (2 * torch.pi) * torch.vstack([freqs, freqs])
        # self.linear.weight = nn.Parameter(new_linear, requires_grad=False)

    def forward(self, input):
        sine_features = torch.sin(2 * torch.pi * (self.freqs.cuda() @ input.t()))
        cosine_features = torch.cos(2 * torch.pi * (self.freqs.cuda() @ input.t()))
        
        # Concatenate sine and cosine features
        return torch.vstack([sine_features, cosine_features]).t()

    # def forward(self, input):
    #     if self.is_first:
    #         x = self.linear(input)

    #     else:
    #         NotImplementedError
    #     return F.tanh(x)
class HybridModel(nn.Module):
    """
    This SIREN version comes from:
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, first_omega_0, hidden_omega_0,
                 bias=True, outermost_linear=True, superposition_w0=True, 
                 **kwargs):
        super().__init__()

        if not isinstance(hidden_features, Sequence):
            hidden_features = [hidden_features] * (hidden_layers + 1)

        activations = kwargs.get('activations', ['sine'] * hidden_layers)
        hidden_idx = 0
        self.net = []
        self.net.append(
            layer_act[activations[hidden_idx]](in_features,
                                               hidden_features[hidden_idx],
                                               bias=bias,
                                               is_first=True,
                                               omega_0=first_omega_0,
                                               **kwargs))

        self.n_layers = hidden_layers + 1
        while hidden_idx < hidden_layers:

            hidden_idx += 1
            self.net.append(
                layer_act[activations[hidden_idx]](hidden_features[hidden_idx - 1],
                                                   hidden_features[hidden_idx],
                                                   is_first=False,
                                                   omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[hidden_idx], out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features[hidden_idx]) / hidden_omega_0,
                    np.sqrt(6 / hidden_features[hidden_idx]) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[hidden_idx], out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def reset_weights(self):
        def reset_sinelayer(m):
            if isinstance(m, SineLayer):
                m.init_weights()
        self.apply(reset_sinelayer)

    def forward(self, coords):
        # allows to take derivative w.r.t. input
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


layer_act = {'sine': SineLayer, 'relu': ReLULayer, 'tanh': TanhLayer, 'ffm': FFMLayer}


if __name__ == '__main__':
    a = SineLayer(2, 8, bias=True, is_first=True, omega_0=30, low_range=20,
                  period=2, bandlimit=60, mode="uniform")
    print(a.linear.weight.shape, a.linear.weight.dtype)
    print("Frequencies sampled uniformly")
    print(a.linear.weight/np.pi)

    b = SineLayer(2, 8, bias=True, is_first=True, omega_0=30,
                  low_range=20, period=2, bandlimit=60, mode="sampling")
    print(b.linear.weight.shape, a.linear.weight.dtype)
    print("Frequencies chosen as a spectral sampling")
    print(b.linear.weight/np.pi)

    c = Siren(in_features=2, hidden_features=[24, 48, 96], hidden_layers=2,
              out_features=3, first_omega_0=60, hidden_omega_0=60)
    print(c)

    d = Siren(in_features=2, hidden_features=[24, 48, 96], hidden_layers=2,
              out_features=3, first_omega_0=60, hidden_omega_0=60,
              low_range=20, period=2, bandlimit=60, mode="sampling")
    print(d)
    print(d.net[0].linear.weight)
