import yaml
from yaml.loader import SafeLoader
import numpy as np
import torch
from scipy.interpolate import griddata
from itertools import product

from mrnet.datasets.pyramids import create_MR_structure
from mrnet.training.optimizer import (OptimizationHandler,
                                      MirrorOptimizationHandler)
from mrnet.datasets.signals import ImageSignal
from mrnet.networks.mrnet import MRFactory

from training.optimizer import (CompressOptimizationHandler,
                                CapacityOptimizationHandler)


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
    elif handler_type == 'compression':
        return CompressOptimizationHandler
    elif handler_type == 'capacity':
        return CapacityOptimizationHandler
    else:
        raise ValueError("Invalid handler_type")


def compression_ratio(model, tol=0.0001):
    total = 0.
    pruned = 0.
    for layer in model.parameters():
        total += torch.sum(torch.ones_like(layer))
        pruned += torch.sum(torch.abs(layer) < tol)
    return (total / (total - pruned)).item()


def space_saving(model, tol=0.0001):
    total = 0.
    pruned = 0.
    for layer in model.parameters():
        total += torch.sum(torch.ones_like(layer))
        pruned += torch.sum(torch.abs(layer) < tol)
    return 1 - ((total - pruned) / total).item()


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


# Only for 1 stage models
def get_database(hyper, train_test=False, percentage_test=0.05):

    base_signal = ImageSignal.init_fromfile(
                        hyper['data_path'],
                        # domain=[-1 + 1 / (hyper['height'] - 1),
                        #         1 - 1 / (hyper['height'] - 1)],
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

    # delta = 1 / (h-1) + (1 - 1 / (h-1)) / h
    # new_data = interpolate_grid_and_image(hyper, base_signal)
    # test_dataset = [ImageSignal(new_data, domain=[-1 + 1/delta, 1 - 1/delta])]
    if train_test:
        base_signal = ImageSignal.init_fromfile(
                            hyper['data_path'],
                            # domain=[-1 + 1/(h-1), 1 - 1/(h-1)],
                            domain=[-1, 1],
                            channels=hyper['channels'],
                            sampling_scheme=hyper['sampling_scheme'],
                            width=w, height=h,
                            batch_size=hyper['batch_size'],
                            color_space=hyper['color_space'],)
        test_mask = ~train_mask
        base_signal.add_mask(test_mask)
    test_dataset = [base_signal]

    return train_dataset, test_dataset


def get_MR_Structure(hyper):
    base_signal = ImageSignal.init_fromfile(
                        hyper['data_path'],
                        domain=[-1 + 1 / (hyper['height'] - 1),
                                1 - 1 / (hyper['height'] - 1)],
                        channels=hyper['channels'],
                        sampling_scheme=hyper['sampling_scheme'],
                        width=hyper['width'], height=hyper['height'],
                        batch_size=hyper['batch_size'],
                        color_space=hyper['color_space'],)
    train = test = create_MR_structure(
        base_signal,
        hyper['max_stages'],
        hyper['filter'],
        hyper['decimation'],
        hyper['pmode']
        )
    return train, test

def init_model(hyper, file_name, device='cpu'):
    model = MRFactory.from_dict(hyper)
    obj = torch.load(file_name)
    model.stages[0].load_state_dict(obj['module0_state_dict'])
    model.stages[0].to(device)
    return model


def interpolate_grid_and_image(hyper, base_signal):
    new_n_height = hyper['height'] - 1
    new_n_width = hyper['width'] - 1

    # Create new grid coordinates
    new_x = np.linspace(-1 + 1 / new_n_height, 1 - 1 / new_n_height,
                        new_n_height)
    new_y = np.linspace(-1 + 1 / new_n_width, 1 - 1 / new_n_width, new_n_width)
    new_coords = list(product(new_x, new_y))
    # Interpolate image
    data_test = griddata(np.array(base_signal.coords),
                         np.array(base_signal.data.view(3,
                                                        -1).permute((1, 0))),
                         new_coords)

    new_data = torch.tensor(data_test).float().reshape(hyper['height'] - 1,
                                                       -1, 3)

    return new_data.permute((2, 0, 1))


# class FFTOptimizationHandler(OptimizationHandler):
#     def __init__(self, model, optimizer, loss_function,
#                  loss_weights, bound, **kwargs):
#         super().__init__(model, optimizer, loss_function, loss_weights)
#         self.bound = (bound / self.get_omega_0())
#         dev = model.stages[-1].middle_layers[0].linear.weight.get_device()
#         device = torch.device(f'cuda:{dev}' if dev >= 0 else 'cpu')
#         self.bound = self.bound.to(device)
#         self.epochs = 0

#     def _post_process(self, loss_dict):

#         loss = sum([loss_dict[key] * self.loss_weights[key]
#                     for key in loss_dict.keys()])

#         loss.backward()
#         self.optimizer.step()

#         for name, weight in self.model.state_dict().items():
#             if 'middle' in name and 'weight' in name:
#                 new_weight = torch.clamp(weight,
#                                          min=-self.bound,
#                                          max=self.bound)
#                 with torch.no_grad():
#                     weight.copy_(new_weight)

#         running_loss = {}
#         for key, value in loss_dict.items():
#             running_loss[key] = (running_loss.get(key, 0.0)
#                                  + value.item())
#         return running_loss


from typing import Sequence
from mrnet.datasets.utils import make_grid_coords, INVERSE_COLOR_MAPPING
from mrnet.datasets.sampler import BatchSampler
import torch
import matplotlib.pyplot as plt
import numpy as np

def log_images(pixels, label, hyper, captions=None, **kw):
    if not isinstance(pixels, Sequence):
        pixels = [pixels]
    if captions is None:
        captions = [None] * len(pixels)
    if isinstance(captions, str):
        captions = [captions]
    if len(pixels) != len(captions):
        raise ValueError("label and pixels should have the same size")
    
    try:
        # TODO: deal with color transform
        color_transform = INVERSE_COLOR_MAPPING[hyper.get(
                                            'color_space', 'RGB')]
        pixels = [color_transform(p.cpu()).clamp(0, 1) 
                for p in pixels]
    except:
        pass
    plt.imshow(pixels[0])
    # plt.title(label)
    plt.axis('off')
    plt.savefig(f"{label}.png", bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def get_prediction(model, test_loader, hyper, device):
    datashape = test_loader.shape[1:]
    
    coords = make_grid_coords(datashape, 
                                *hyper['domain'],
                                len(datashape))
    pixels = []
    for batch in BatchSampler(coords, 
                                hyper['batch_size'], 
                                drop_last=False):
        batch = torch.stack(batch)
        output_dict = model(batch.to(device))
        pixels.append(output_dict['model_out'].detach().cpu())
        
    pixels = torch.concat(pixels)
    return pixels

def log_prediction(model, test_loader, hyper, label, device):

    pixels = get_prediction(model, test_loader, hyper, device)
    datashape = test_loader.shape[1:]

    pred_pixels = pixels.reshape((*datashape, hyper['channels']))
    if hyper.get('normalize_view', False):
            vmax = torch.max(pred_pixels)
            vmin = torch.min(pred_pixels)
            pred_pixels = (pred_pixels - vmin) / (vmax - vmin)
    log_images(pred_pixels, label, hyper, category='pred')