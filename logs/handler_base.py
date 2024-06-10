
import os
from matplotlib import pyplot as plt
from mrnet.logs.handler import (make_grid_coords, BatchSampler,
                                rgb_to_grayscale, gradient)
import torch
import numpy as np
from PIL import Image
from skimage.filters import sobel


def log_metrics(gt, pred, name):
    mse = torch.mean((gt - pred)**2)
    psnr = (10 * torch.log10(1 / mse)).item()
    np.savetxt(name + '/psnr.csv', [psnr], delimiter="\n")
    return psnr


def log_data(model, train_loader, test_loader, device, batch_size, channels,
             name):

    directory = '/home/diana/mrnet-compression/mrnet/runs/logs/bacon/' + name
    if not os.path.exists(directory):
        os.makedirs(directory)

    pixels = []
    gt = []
    color_space = 'RGB'
    for batch in test_loader:
        X, gt_dict = batch['c0']
        X['coords'] = X['coords'].to(device)
        X['coords'] = X['coords'].requires_grad_(True)
        pred = model(X['coords'])[0]
        pixels.append(pred.detach().cpu())
        gt.append(gt_dict['d0'].detach().cpu())
    pixels = torch.concat(pixels)
    gt = torch.concat(gt)
    psnr = log_metrics(gt, pixels, directory)

    datashape = test_loader.shape[1:]
    coords = make_grid_coords(datashape,
                              *[-1, 1],
                              len(datashape))
    pixels = []
    grads = []
    color_space = 'RGB'
    for batch in BatchSampler(coords, batch_size,
                              drop_last=False):
        batch = torch.stack(batch)
        batch.requires_grad = True
        pred = model(batch.to(device))[0]
        pixels.append(pred.detach().cpu())
        value = {'output': pred}
        if color_space == 'YCbCr':
            value = value[:, 0:1]
        elif color_space == 'RGB':
            value = rgb_to_grayscale(value['output'])
        grads.append(gradient(value,
                              pred
                              ).detach().cpu())

    pixels = torch.concat(pixels)
    pred_pixels = pixels.reshape((*datashape, channels))
    vmax = torch.max(pred_pixels)
    vmin = torch.min(pred_pixels)
    pred_pixels = (pred_pixels - vmin) / (vmax - vmin)
    # img = Image.fromarray((pred_pixels.numpy()).astype(np.uint8))
    plt.imsave(directory + '/Prediction.png', pred_pixels.numpy())

    if color_space == 'YCbCr':
        gray_pixels = pred_pixels[..., 0]
    elif color_space == 'RGB':
        gray_pixels = rgb_to_grayscale(pred_pixels).squeeze(-1)
    elif color_space == 'L':
        gray_pixels = pred_pixels.squeeze(-1)
    else:
        raise ValueError(f"Invalid color space: {color_space}")

    log_fft(gray_pixels, directory + '/FFT Prediction', category='pred')

    grads = torch.concat(grads)
    grads = grads.reshape((*datashape, 2))
    grad_img = log_gradmagnitude(grads, directory + '/Gradient Magnitude Pred',
                                 category='pred')
    gt_img = train_loader.data.permute((1, 2, 0))
    psnr_grad = log_grad_psnr(directory, gt_img.numpy(), grad_img)
    return [psnr, psnr_grad]


def log_gradmagnitude(grads: torch.Tensor, label: str, **kw):
    mag = np.hypot(grads[:, :, 0].squeeze(-1).numpy(),
                   grads[:, :, 1].squeeze(-1).numpy())
    gmin, gmax = np.min(mag), np.max(mag)
    img = Image.fromarray(255 * (mag - gmin) / (gmax - gmin)).convert('L')

    plt.imsave(f'{label}.png', img, cmap='gray')
    return img


def log_fft(pixels: torch.Tensor, label: str,
            captions=None, **kw):
    '''Assumes a grayscale version of the image'''

    fft_pixels = torch.fft.fft2(pixels)
    fft_shifted = torch.fft.fftshift(fft_pixels).numpy()
    magnitude = np.log(1 + abs(fft_shifted))
    # normalization to visualize as image
    vmin, vmax = np.min(magnitude), np.max(magnitude)
    magnitude = (magnitude - vmin) / (vmax - vmin)
    img = Image.fromarray((magnitude * 255).astype(np.uint8))
    plt.imsave(f'{label}.png', img, cmap='gray')


def log_grad_psnr(directory, gt, pred_grad):
    gray_gt = np.dot(gt[...,:3], [0.2990, 0.5870, 0.1140])
    # Compute gradient using the Sobel operator
    gradient_x = sobel(gray_gt, axis=1)
    gradient_y = sobel(gray_gt, axis=0)

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
    np.savetxt(directory + '/grad_psnr.csv', [psnr_value], delimiter="\n")
    return psnr_value