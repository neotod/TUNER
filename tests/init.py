import sys
import unittest
import torch
import yaml
from yaml.loader import SafeLoader
sys.path[0] = sys.path[0].rstrip("/tests")

from networks.siren_mrnet import SineLayer  #, Siren


def load_hyperparameters(config_path):
    with open(config_path) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
            hyper['channels'] = hyper['out_features']

    return hyper


hyper = load_hyperparameters("configs/config_init.yml")


class TestdfGrad(unittest.TestCase):

    def test_shape_and_integer_freqs(self):
        in_feats, out_feats, w0, period = 2, 10, 30, 2
        layer = SineLayer(in_feats,
                          out_feats,
                          omega_0=w0,
                          is_first=True,
                          period=period)
        constant = 2 * torch.pi / period
        input_freqs = layer.linear.weight / constant

        # Check all frequencies are below w0
        self.assertTrue(torch.max(input_freqs) <= w0)

        # Check shape of layer
        self.assertTrue(input_freqs.shape[1] == in_feats)
        self.assertTrue(input_freqs.shape[0] == out_feats)

        # Check the frequencies are frozen
        self.assertTrue(not input_freqs.requires_grad)

        # Check the frequencies are integers
        torch.allclose(input_freqs - input_freqs.int(),
                       torch.zeros_like(input_freqs))

    def test_different_dimensions(self):
        in_feats, out_feats, w0, period = 3, 10, 30, 2
        layer = SineLayer(in_feats,
                          out_feats,
                          omega_0=w0,
                          is_first=True,
                          period=period)
        constant = 2 * torch.pi / period
        input_freqs = layer.linear.weight / constant

        # Check all frequencies are below w0
        self.assertTrue(torch.max(input_freqs) <= w0)

        # Check shape of layer
        self.assertTrue(input_freqs.shape[0] == out_feats)
        self.assertTrue(input_freqs.shape[1] == in_feats)

        # Check the frequencies are frozen
        self.assertTrue(not input_freqs.requires_grad)

        # Check the frequencies are integers
        torch.allclose(input_freqs - input_freqs.int(),
                       torch.zeros_like(input_freqs))


if __name__ == '__main__':
    unittest.main()
