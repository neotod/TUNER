import csv
import os
import time
import torch
import numpy as np
from skimage import io
from skimage.filters import sobel

from mrnet.logs.logger import LocalLogger
from mrnet.logs.handler import (ImageHandler, BatchSampler,
                                rgb_to_grayscale, make_grid_coords)
from mrnet.training.listener import TrainingListener
from mrnet.networks.mrnet import MRNet

from utils import get_database


class TrackLocalLogger(LocalLogger):

    def __init__(self, project: str,
                 name: str,
                 hyper: dict,
                 basedir: str,
                 **kwargs):
        hyper['stage'] = 0
        super().__init__(project, name, hyper,
                         basedir, **kwargs)
        self.subpaths['track'] = 'track'


class TrackImageHandler(ImageHandler):

    def __init__(self, project: str, name: str, hyper: dict,
                 basedir: str, logger=None) -> None:
        super().__init__(hyper, logger)
        self.logger = TrackLocalLogger(project, name, hyper, basedir)

    def log_grad_psnr(self):

        gt = self.logger.savedir + '/gt/ground_truth01.png'
        grad = self.logger.savedir + '/pred/gradient_magnitude_pred01.png'
        gt_img = io.imread(gt, as_gray=True)
        pred_grad = io.imread(grad, as_gray=True)

        # Compute gradient using the Sobel operator
        gradient_x = sobel(gt_img, axis=1)
        gradient_y = sobel(gt_img, axis=0)

        # Compute gradient magnitude
        gt_grad_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Normalize the gradient magnitude to the range [0, 1]
        gt_grad_magnitude_normalized = \
            gt_grad_magnitude / np.max(gt_grad_magnitude)

        pred_grad = np.array(pred_grad)

        squared_diff = (pred_grad.astype(np.float16)/255 -
                        gt_grad_magnitude_normalized.astype(np.float16)) ** 2
        mse_value = np.mean(squared_diff)
        max_pixel = 1.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
        log_dict = {'grad_psnr': psnr_value}
        save_path = os.path.join(self.logger.savedir, 'grad_psnr.csv')
        file_exists = os.path.isfile(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_dict)

    def log_track(self, model, test_loader, epoch, device):
        datashape = test_loader.shape[1:]
        coords = make_grid_coords(datashape,
                                  *self.hyper['domain'],
                                  len(datashape))
        pixels = []
        color_space = self.hyper['color_space']
        for batch in BatchSampler(coords,
                                  self.hyper['batch_size'],
                                  drop_last=False):
            batch = torch.stack(batch)
            output_dict = model(batch.to(device))
            pixels.append(output_dict['model_out'].detach().cpu())
            value = output_dict['model_out']
            if color_space == 'YCbCr':
                value = value[:, 0:1]
            elif color_space == 'RGB':
                value = rgb_to_grayscale(value)

        pixels = torch.concat(pixels)
        pred_pixels = pixels.reshape((*datashape, self.hyper['channels']))
        if self.hyper.get('normalize_view', False):
            vmax = torch.max(pred_pixels)
            vmin = torch.min(pred_pixels)
            pred_pixels = (pred_pixels - vmin) / (vmax - vmin)
        name = 'Pred' + str(epoch)
        self.logger.log_images(pred_pixels, name, category='track', fnames='')

        if color_space == 'YCbCr':
            gray_pixels = pred_pixels[..., 0]
        elif color_space == 'RGB':
            gray_pixels = rgb_to_grayscale(pred_pixels).squeeze(-1)
        elif color_space == 'L':
            gray_pixels = pred_pixels.squeeze(-1)
        else:
            raise ValueError(f"Invalid color space: {color_space}")
        name = 'FFT' + str(epoch)
        self.log_fft(gray_pixels, name, category='track', fnames='')

        return pixels
    
    def log_partial_fft(self, model, test_loader, epoch, device):
        datashape = test_loader.shape[1:]
        coords = make_grid_coords(datashape,
                                  *self.hyper['domain'],
                                  len(datashape))
        pixels = []
        color_space = self.hyper['color_space']
        for batch in BatchSampler(coords,
                                  self.hyper['batch_size'],
                                  drop_last=False):
            batch = torch.stack(batch)
            hidden_neurons = mean_neuron_activation(model, batch, device)
            pixels.append(hidden_neurons.detach().cpu())
            value = hidden_neurons
            if color_space == 'YCbCr':
                value = value[:, 0:1]
            elif color_space == 'RGB':
                value = rgb_to_grayscale(value)

        pixels = torch.concat(pixels)
        pred_pixels = pixels.reshape((*datashape, -1)).mean(dim=-1)
        if self.hyper.get('normalize_view', False):
            vmax = torch.max(pred_pixels)
            vmin = torch.min(pred_pixels)
            pred_pixels = (pred_pixels - vmin) / (vmax - vmin)
        name = 'Activation'
        self.logger.log_images(pred_pixels, name, category='pred', fnames='')

        name = 'Activation FFT'
        self.log_fft(pred_pixels, name, category='pred', fnames='')

        return pixels

def mean_neuron_activation(model, batch, device, i=-1):
    model.eval()
    first_layer = model.stages[0].first_layer.to(device)
    neurons = torch.sin(first_layer(batch.to(device)))
    for layer in model.stages[0].middle_layers[:i]:
        neurons = torch.sin(layer(neurons) * layer.omega_0)
    return neurons



class TrackTrainingListener(TrainingListener):

    def __init__(self, project: str,
                 name: str,
                 hyper: dict,
                 basedir: str,
                 entity=None,
                 config=None,
                 settings=None) -> None:

        super().__init__(project, name, hyper, basedir, entity, config,
                         settings)
        self.handler = TrackImageHandler(project, name, hyper, basedir)

    def on_stage_start(self, current_model, 
                       stage_number, updated_hyper=None):
        if updated_hyper:
            for key in updated_hyper:
                self.hyper[key] = updated_hyper[key]

        logger = TrackLocalLogger(self.project,
                                  self.name,
                                  self.hyper,
                                  self.basedir,
                                  stage=stage_number, 
                                  entity=self.entity, 
                                  config=self.config, 
                                  settings=self.settings)
        logger.prepare(current_model)
        self.handler.logger = logger

    def on_train_start(self):
        self.epoch = 0

    def on_epoch_finish(self, current_model, epochloss):

        self.epoch += 1
        if self.epoch % 20 == 0:
            _, test_dataset = get_database(self.hyper)
            device = self.hyper.get('eval_device', 'cpu')
            self.handler.log_track(current_model,
                                   test_dataset[0],
                                   self.epoch,
                                   device)
        self.handler.log_losses(epochloss)

    def on_stage_trained(self, current_model: MRNet,
                         train_loader,
                         test_loader):
        device = self.hyper.get('eval_device', 'cpu')
        current_stage = current_model.n_stages()
        current_model.eval()
        current_model.to(device)
        
        start_time = time.time()
        
        self.handler.log_chosen_frequencies(current_model)
        gt = self.handler.log_groundtruth(test_loader,
                                          train_loader,
                                          stage=current_stage)
        pred = self.handler.log_prediction(current_model, test_loader, device)
        self.handler.log_metrics(gt.cpu(), pred.cpu())
        self.handler.log_extrapolation(current_model,
                                       test_loader,
                                       device)
        # self.handler.log_partial_fft(current_model, test_loader, self.epoch, device)
        self.handler.log_zoom(current_model, test_loader, device)
        self.handler.log_grad_psnr()
        # TODO: check for pointcloud data
        print(f"[Logger] All inference done in {time.time() - start_time}s on {device}")
        current_model.train()
        current_model.to(self.hyper['device'])
        self.handler.log_model(current_model)
        self.handler.finish()

    def on_batch_finish(self, batchloss):
        pass

    def on_train_finish(self, trained_model, total_epochs):
        print(f'Training finished after {total_epochs} epochs')
