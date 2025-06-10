import yaml
from yaml.loader import SafeLoader
import torch

from mrnet.training.optimizer import (OptimizationHandler,
                                      MirrorOptimizationHandler)
from mrnet.datasets.signals import ImageSignal


def load_hyperparameters(config_path):
    with open(config_path) as f:
        hyper = yaml.load(f, Loader=SafeLoader)
        if isinstance(hyper['batch_size'], str):
            hyper['batch_size'] = eval(hyper['batch_size'])
        if hyper.get('channels', 0) == 0:
            hyper['channels'] = hyper['out_features']

    return hyper


def get_optim_handler(handler_type):
    if handler_type == 'regular':
        return OptimizationHandler
    elif handler_type == 'mirror':
        return MirrorOptimizationHandler
    else:
        raise ValueError("Invalid handler_type")

def create_clamps(clamp_vals, clamp_limits, init_freqs, device='cpu'):
    assert len(clamp_vals) == len(clamp_limits) + 1
    clamps = torch.empty(len(init_freqs), device=device)
    for idx, (fx, fy) in enumerate(init_freqs):
        max_freq = max(abs(fx), abs(fy))
        max_idx = sum(torch.tensor(clamp_limits, device=device) < max_freq)
        clamps[idx] = clamp_vals[max_idx]
    return clamps.to(device)


def create_clamps_by_freq(init_freqs, const):
    clamps = const / torch.max(init_freqs, dim=1)[0]
    return clamps


def mse_loss(output_dict, gt_dict, **kwargs):
    device = kwargs.get('device', 'cpu')
    pred: torch.Tensor = output_dict['model_out']
    pred = pred['output']
    gt = gt_dict['d0'].to(device)
    loss_dict = {'d0': torch.nn.functional.mse_loss(pred, gt, device)}
    return loss_dict


def construct_bound(model, bounds, limit_lf, period):
    # bounds = [c_L, c_H]
    first_weight = model.net[0].linear.weight.clone().detach()
    freqs = (period / 2 * torch.pi) * first_weight
    max_1d_freq = torch.max(freqs, dim=1).values
    mask = max_1d_freq > limit_lf
    bound = torch.ones_like(max_1d_freq)
    bound[mask] = bounds[1]
    bound[~mask] = bounds[0]
    return bound


def get_forward(model, hyper, layer):
    def forward(input):
        W = (torch.tanh(hyper['hidden_omega_0'] / 5 *
                model.net[layer].linear.weight) * model.net[layer].bound)
        x = (input @ W.T + hyper['hidden_omega_0'] * model.net[layer].linear.bias)
        return torch.sin(x)
    return forward


# Only for 1 stage models
def get_database(hyper, train_test=False, percentage_test=0.05, **kw):

    base_signal = ImageSignal.init_fromfile(
                        hyper['data_path'],
                        domain=[-1, 1],
                        channels=hyper['channels'],
                        sampling_scheme=hyper['sampling_scheme'],
                        width=hyper['width'], height=hyper['height'],
                        batch_size=hyper['batch_size'],
                        color_space=hyper['color_space'],)
    if hyper['width'] == 0:
        hyper['width'] = base_signal.shape[-1]
    if hyper['height'] == 0:
        hyper['height'] = base_signal.shape[-1]
    h, w = hyper['height'], hyper['width']

    if train_test:
        N_pts_test = int(h * w * percentage_test)
        train_mask = torch.ones((w, h), dtype=torch.bool)
        idx_test = torch.arange(0, h * w)[torch.randperm(h * w)][:N_pts_test]
        train_mask.view(-1)[idx_test] = False
        base_signal.add_mask(train_mask)
    train_dataset = [base_signal]

    w = kw.get('width_test', w)
    h = kw.get('height_test', h)
    base_signal = ImageSignal.init_fromfile(
                        hyper['data_path'],
                        domain=[-1, 1],
                        channels=hyper['channels'],
                        sampling_scheme=hyper['sampling_scheme'],
                        width=w, height=h,
                        batch_size=hyper['batch_size'],
                        color_space=hyper['color_space'],)
    if train_test:
        test_mask = ~train_mask
        base_signal.add_mask(test_mask)
    test_dataset = [base_signal]

    return train_dataset, test_dataset
