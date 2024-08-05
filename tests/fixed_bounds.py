import sys
import unittest
import torch
from pathlib import Path
sys.path[0] = sys.path[0].rstrip("/tests")

from mrnet.training.trainer import MRTrainer
from mrnet.training.listener import TrainingListener

from networks.mrnet import MRFactory
from training.optimizers import ClampOptimizationHandler
from utils import (create_clamps, load_hyperparameters,
                   get_database)


RANDOM_SEED = 777


def train_model(bounds, hyper=None):

    if hyper is None:
        hyper = load_hyperparameters("configs/config_init.yml")

    mrmodel = MRFactory.from_dict(hyper)

    first_weight = mrmodel.stages[0].first_layer.linear.weight
    freqs = first_weight * hyper['period'] / (2 * torch.pi)
    bounds_1hl = create_clamps(bounds[0],
                               hyper['block_limits'],
                               freqs)
    c = [bounds_1hl, bounds[1:]] if len(bounds) > 1 else [bounds_1hl]

    def optim_handler(model, optimizer, loss_function, loss_weights):
        return ClampOptimizationHandler(
            model, optimizer, loss_function, loss_weights, c
            )
    print("Model: ", type(mrmodel))
    logger = TrainingListener(
        'Test',
        f"test_bounds_{'_'.join(map(str, bounds))}",
        hyper,
        Path(hyper.get("log_path", "runs")),
    )
    train_dataset, test_dataset = get_database(hyper, train_test=False)
    mrtrainer = MRTrainer.init_from_dict(
        mrmodel, train_dataset, test_dataset, logger, hyper,
        optim_handler=optim_handler
    )
    mrtrainer.train(hyper["device"])
    return mrmodel


def abs_dist(a):
    return torch.max(torch.abs(a), axis=-1)[0]


class TestdfGrad(unittest.TestCase):

    # Test if the bounds over low/high frequencies work as expected
    def test_bounds_for_low_high_freqs(self):
        hyper = load_hyperparameters("configs/config_init.yml")
        hyper['hidden_layers'] = 1
        bounds = [[1., .2]]
        # Check all frequencies are below w0
        model = train_model(bounds, hyper)
        first_weight = model.stages[0].first_layer.linear.weight
        freqs = first_weight * hyper['period'] / (2 * torch.pi)
        low_f_mask = abs_dist(freqs) < hyper['block_limits'][0]
        w0 = model.stages[0].middle_layers[0].omega_0
        Wt = model.stages[0].middle_layers[0].linear.weight * w0
        low_b = torch.all(Wt[:, low_f_mask] < bounds[0][0]).item()
        high_b = torch.all(Wt[:, ~low_f_mask] < bounds[0][1]).item()
        self.assertTrue(low_b and high_b)

    # Test that bounds are bounding the weights
    def test_deep_models(self):
        hyper = load_hyperparameters("configs/config_init.yml")
        hyper['hidden_layers'] = 2
        hyper['hidden_features'] = [[120, 120, 120]]
        bounds = [[1., .2], 0.1]
        # Check all frequencies are below w0
        model = train_model(bounds, hyper)
        w0 = model.stages[0].middle_layers[1].omega_0
        Wt = model.stages[0].middle_layers[1].linear.weight.T * w0
        bounded = [abs_dist(col) < bounds[1] for col in Wt]
        self.assertTrue(all(bounded))


if __name__ == '__main__':
    unittest.main()
