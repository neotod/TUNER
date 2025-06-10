
import csv
import os
from datetime import date, datetime
from matplotlib import pyplot as plt
from mrnet.logs.handler import (make_grid_coords, BatchSampler,
                                rgb_to_grayscale)
import torch
import numpy as np
from PIL import Image
from skimage.filters import sobel
import yaml


RESIZING_FILTERS = {
    'nearest': Image.NEAREST,
    'linear': Image.BILINEAR,
    'cubic': Image.BICUBIC,
}

DIR = os.getcwd()


INVERSE_COLOR_MAPPING = {
    'RGB': lambda x: x,
    'L': lambda x: x,
}


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


def log_metrics(gt, pred, name):
    mse = torch.mean((gt - pred)**2)
    psnr = (10 * torch.log10(1 / mse)).item()
    np.savetxt(name + '/psnr.csv', [psnr], delimiter="\n")
    return psnr


def log_data(model, train_loader, test_loader, hyper,
             name, **kw):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    directory = DIR + f'/runs/logs/{timestamp}_{name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    device = 'cuda:0' if hyper["device"] == 'cuda' else 'cpu'
    batch_size = hyper["batch_size"]
    channels = hyper["channels"]
    
    # Save model
    torch.save(model.state_dict(), directory + '/model.pth')

    # Save hyperparameters
    hypercontent = yaml.dump(hyper)
    with open(os.path.join(directory, "hyper.yml"), "w") as hyperfile:
        hyperfile.write(hypercontent)


    # Log loss across training epochs
    if 'loss' in kw:
        loss = kw['loss']
        with open(directory + '/loss.csv', 'w') as f:
            for i, item in enumerate(loss):
                f.write("%s\n" % item)

    # Log PSNR
    pixels = []
    gt = []
    color_space = 'RGB'
    for batch in train_loader:
        X, gt_dict = batch['c0']
        # X['coords'] = X['coords'].requires_grad_(True)
        pred = model(X['coords'].to(device))[0]
        pixels.append(pred.detach().cpu())
        gt.append(gt_dict['d0'].detach().cpu())
    pixels = torch.concat(pixels)
    gt = torch.concat(gt)
    psnr = log_metrics(gt, pixels, directory)

    # Log input frequencies
    log_chosen_frequencies(model, hyper, directory)
    
    datashape = test_loader.shape[1:]
    coords = make_grid_coords(datashape,
                              *hyper["domain"],
                              len(datashape))

    pixels = []
    grads = []
    color_space = hyper["color_space"]
    for batch in BatchSampler(coords, batch_size,
                              drop_last=False):
        batch = torch.stack(batch)
        pred, coords = model(batch.to(device))
        pixels.append(pred.detach().cpu())
        value = {'output': pred}
        if color_space == 'YCbCr':
            value = value[:, 0:1]
        elif color_space == 'RGB':
            value = rgb_to_grayscale(value['output'])
        grads.append(gradient(value,
                              coords,
                              ).detach().cpu())
    grads = torch.concat(grads)
    grads = grads.reshape((*datashape, 2))
    grad_img = log_gradmagnitude(grads, directory + '/Gradient Magnitude Pred',
                                 category='pred')

    pixels = torch.concat(pixels)
    pred_pixels = pixels.reshape((*datashape, channels))
    array = (pred_pixels.clamp(0, 1).numpy() * 255).astype(np.uint8)
    # Image.fromarray(array).save(directory + '/Prediction.png')
    plt.imsave(directory + '/Prediction.png', array)

    if color_space == 'YCbCr':
        gray_pixels = pred_pixels[..., 0]
    elif color_space == 'RGB':
        gray_pixels = rgb_to_grayscale(pred_pixels).squeeze(-1)
    elif color_space == 'L':
        gray_pixels = pred_pixels.squeeze(-1)
    else:
        raise ValueError(f"Invalid color space: {color_space}")
    fft_pred = log_fft(gray_pixels, directory + '/FFT Prediction', category='pred')
    if kw.get('bandlimit_metric', None) is not None:
        bandlimit = kw['bandlimit_metric']
        res = fft_pred.shape[0]
        mask = np.ones_like(fft_pred)
        b = int((bandlimit * 1.1 + 1) / 2)
        mask[res // 2 - b: res // 2 + b, res // 2 - b: res // 2 + b] = 0.
        freq_error = np.copy(fft_pred)
        freq_error[mask == 0] = 0
        mse_error = np.mean((np.zeros_like(mask) - freq_error) ** 2)
        # plt.imshow(freq_error, cmap='gray')
        # plt.show()
        # plt.axis('off')
        with open(f"{directory}/mse_band{bandlimit}.txt", "w") as f:
            f.write(f"{mse_error}")
            f.close()

    gt_img = test_loader.data.permute((1, 2, 0))
    log_zoom(hyper, model, test_loader, directory, device)

    psnr_grad = log_grad_psnr(directory, gt_img.numpy(), grad_img)
    psnr_test = 0.
    if test_loader is not None:
        psnr_test = log_psnr_test(directory, gt_img, pixels, test_loader[0]['c0'][0]['idx'])

    if kw.get('bandlimit_metric', None) is not None:
        print("Train PSNR: {:.2f}  -  Test PSNR: {:.2f}  -  Grad PSNR: {:.2f}  -   Error FFT: {:.2f}".format(psnr, psnr_test, psnr_grad, mse_error))
    else:
        print("Train PSNR: {:.2f}  -  Test PSNR: {:.2f}  -  Grad PSNR: {:.2f}".format(psnr, psnr_test, psnr_grad))
    print(f'Results saved in {directory}\n')


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
    return (magnitude * 255).astype(np.uint8)

def log_zoom(hyper, model, test_loader, save_dir, device):
    w, h = test_loader.shape[1:]
    domain = hyper.get('domain', [-1, 1])
    zoom = hyper.get('zoom', [])
    for zoom_factor in zoom:
        start, end = domain[0]/zoom_factor, domain[1]/zoom_factor
        zoom_coords = make_grid_coords((w, h), start, end, dim=2)
        with torch.no_grad():
            output = []
            for batch in BatchSampler(zoom_coords, hyper['batch_size'], drop_last=False):
                batch = torch.stack(batch).to(device)
                output.append(model(batch)[0])
            pixels = torch.concat(output)

        pixels = pixels.cpu().view((h, w, hyper['channels']))
        if (hyper['channels'] == 1 
            and hyper['loss_weights']['d0'] == 0):
            vmin = torch.min(test_loader.data)
            vmax = torch.max(test_loader.data)
            pmin, pmax = torch.min(pixels), torch.max(pixels)
            pixels = (pixels - pmin) / (pmax - pmin)
            pixels = pixels * vmax #(vmax - vmin) + vmin
        
        # center crop
        cropsize = int(w // zoom_factor)
        left, top = (w - cropsize), (h - cropsize)
        right, bottom = (w + cropsize), (h + cropsize)
        crop_rectangle = tuple(np.array([left, top, right, bottom]) // 2)
        gt_pixels = test_loader.data.permute((1, 2, 0))
        
        color_space = hyper['color_space']
        color_transform = INVERSE_COLOR_MAPPING[color_space]
        pixels = color_transform(pixels).clamp(0, 1)
        gt_pixels = color_transform(gt_pixels).clamp(0, 1)

        pixels = (pixels.clamp(0, 1) * 255).squeeze(-1).numpy().astype(np.uint8)
        
        images = [Image.fromarray(pixels)]
        # captions = [f'{zoom_factor}x Reconstruction (Ours)']
        fnames = [f'zoom_{zoom_factor}x_ours']
        gt_pixels = (gt_pixels * 255).squeeze(-1).numpy().astype(np.uint8)
        # cropped = Image.fromarray(gt_pixels).crop(crop_rectangle)
        # for filter in hyper.get('zoom_filters', ['linear']):
        #     resized = cropped.resize((w, h), RESIZING_FILTERS[filter])
        #     images.append(resized)
        #     captions.append(f"{zoom_factor}x Baseline - {filter} interpolation")
        #     fnames.append(f'zoom_{zoom_factor}x_{filter}')
            
        # self.logger.log_images(images, f"Zoom {zoom_factor}x", captions, 
        #                             fnames=fnames, category='zoom')
        # return images, zoom_factor, fnames
        # pixels, captions = super().log_images(pixels, label, captions)
        path = save_dir
        pixels = images

        for i, image in enumerate(pixels):
            filepath = os.path.join(path, fnames[i] + ".png")
            try:
                image.save(filepath)
            # if it is not a PIL Image
            except AttributeError:
                array = image.squeeze(-1).numpy()
                if array.dtype != np.uint8 and np.max(array) <= 1.0:
                    array = (array * 255).astype(np.uint8)
                Image.fromarray(array).save(filepath)

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

def log_psnr_test(save_dir, gt, pred, mask):
        gt = gt.view(1, -1, gt.size(2)).squeeze()
        test_gt = gt[mask]
        test_pred = pred[mask]
        mse = torch.mean((test_gt - test_pred)**2)
        psnr = 10 * torch.log10(1 / mse)
        log_dict = {'test_psnr': psnr.item()}
        save_path = os.path.join(save_dir, 'test_psnr.csv')
        file_exists = os.path.isfile(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_dict)
        return psnr.item()

def log_chosen_frequencies(model, hyper, save_dir):
    try:
        frequencies = model.net[0].linear.weight.detach().cpu().numpy()
        frequencies = (frequencies * hyper["period"] 
                        / (2 * np.pi)).astype(np.int32)
    except:
        frequencies = model.net[0].freqs.detach().cpu().numpy().astype(np.int32)
    h, w = hyper['width'], hyper['height']
    frequencies = frequencies + np.array((h//2, w//2))
    img = Image.new('L', (h, w))
    # print(frequencies.max())

    # exit()
    for f in frequencies:
        img.putpixel(f, 255)
    
    # self.logger.log_images([img], "Chosen Frequencies", category='etc')

    # pixels, captions = super().log_images(pixels, label, captions)
    path = save_dir

    image = img
    filename = "Chosen Frequencies"    
    filepath = os.path.join(path, filename + ".png")
    try:
        image.save(filepath)
    # if it is not a PIL Image
    except AttributeError:
        array = image.squeeze(-1).numpy()
        if array.dtype != np.uint8 and np.max(array) <= 1.0:
            array = (array * 255).astype(np.uint8)
        Image.fromarray(array).save(filepath)