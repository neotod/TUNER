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


def n_cartesian_freqs(high_list):
    # half of cartesian product of list [*high_list, *-high_list]
    # where high_list is a list of positive integers, excluding (0, 0)
    if 0 in high_list:
        return ((2 * len(high_list)) ** 2) / 2
    else:
        return ((2 * len(high_list) + 1) ** 2) / 2


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
        print(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mode = mode

        self.__set_bandlimit(kwargs.get('bandlimit', 0))
        self.__set_low_range(kwargs.get('low_range', 0))
        self.__set_perc_low_freqs(kwargs.get('perc_low_freqs', 0.5))

        self.period = period
        self.bound = nn.Parameter(torch.tensor(0.2))  # UNDO
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
                assert self.out_features - self.in_features > 0, f"""
                        Must initialize with at least as many frequencies
                        ({self.out_features}) as input dimensions
                        ({self.in_features})."""
                n_freqs = self.out_features - self.in_features
                rng = np.random.default_rng(RANDOM_SEED)
                possible_low_frequencies = cartesian_product(
                    np.arange(0, self.__low_range + 1),
                    np.arange(-self.__low_range, self.__low_range + 1))
                high_1d_freqs = self.list_1d_high_freqs()
                if self.in_features == 2:
                    possible_high_frequencies = cartesian_product(
                        np.array(high_1d_freqs),
                        np.array(list(set(high_1d_freqs +
                                          [-i for i in high_1d_freqs]))))

                else:
                    possible_high_frequencies = cartesian_product(
                        *(self.in_features *
                          [np.array(list(set([*-high_1d_freqs,
                                              *high_1d_freqs])))])
                        )
                if discarded_freqs:
                    possible_high_frequencies = np.array(list(
                        set(tuple(map(tuple, possible_high_frequencies))) -
                        set(used_weights)
                    ))
                chosen_low_frequencies = torch.from_numpy(
                    rng.choice(possible_low_frequencies,
                               int(np.ceil(self.__perc_low_freqs * n_freqs)),
                               False))
                chosen_high_frequencies = torch.from_numpy(
                    rng.choice(possible_high_frequencies,
                               int(np.floor((1 - self.__perc_low_freqs) *
                                            n_freqs)),
                               False))

                chosen_frequencies = torch.cat(
                    (torch.eye(self.in_features),  # initialize with basis
                     chosen_low_frequencies,
                     chosen_high_frequencies))
                self.linear.weight = nn.Parameter(
                    chosen_frequencies.float() * 2 * torch.pi / self.period)
                # first layer will not be updated during training
                self.linear.weight.requires_grad = False

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
        perc_high_freqs = n_cartesian_freqs(high_list) / self.out_features
        while perc_high_freqs < 1 - self.__perc_low_freqs:
            high_list.append(0)
            perc_high_freqs = n_cartesian_freqs(high_list) / self.out_features
        number_high_freqs = len(high_list) - 1
        step = (self.__bandlimit - self.__low_range) // number_high_freqs
        high_list = [0] + [
            self.__low_range + i * step for i in range(number_high_freqs)
            ]
        high_list[-1] = self.__bandlimit
        return high_list

    def __get_bandlimit(self):
        return self.__bandlimit

    def __get_low_range(self):
        return self.__low_range

    def __get_perc_low_freqs(self):
        return self.__perc_low_freqs

    def __set_bandlimit(self, var):
        b = int(var)
        if b > 0:
            self.__bandlimit = b
        else:
            self.__bandlimit = int(self.omega_0)
            warn(f"""Bandlimit {b} must be a positive integer.
                 Setting bandlimit to {self.__bandlimit}.""")

    def __set_low_range(self, var):
        low_limit = int(var)
        if low_limit > 0 and low_limit < self.__bandlimit:
            self.__low_range = var
        else:
            self.__low_range = self.__bandlimit // 3
            warn(f"""Low range {low_limit} must be in [0, {self.__bandlimit}].
            Setting bandlimit to {self.__low_range}.""")

    def __set_perc_low_freqs(self, var):
        if var > 0 and var <= 1:
            self.__perc_low_freqs = var
        else:
            self.__perc_low_freqs = 0.5
            warn(f"""Percentage of low frequencies {var} must be in [0, 1].
            Setting bandlimit to {self.__perc_low_freqs}.""")


class Siren(nn.Module):
    """
    This SIREN version comes from:
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, first_omega_0, hidden_omega_0,
                 bias=True, outermost_linear=False, superposition_w0=True, 
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
        while hidden_idx < hidden_layers - 1:
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
