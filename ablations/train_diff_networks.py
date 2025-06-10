import csv
import os
import torch
import torch.nn.functional as F

from taming.training.handler_base import log_data
from utils import (load_hyperparameters, get_database, mse_loss,
                   construct_bound)

from networks.siren import HybridModel


def mse_loss(output_dict, gt_dict, **kwargs):
    pred: torch.Tensor = output_dict['model_out']
    pred = pred['output']
    gt = gt_dict['d0'].to(device)
    loss_dict = {'d0': F.mse_loss(pred, gt, device)}
    return loss_dict


RANDOM_SEED = 777

if __name__ == "__main__":

    torch.manual_seed(RANDOM_SEED)
    image_paths = ['data/div2k/test_data/' + fname for fname in os.listdir('./data/div2k/test_data/')]
    hyper = load_hyperparameters("configs/config_init.yml")
    device = 'cuda:0' if hyper['device'] == 'cuda' else 'cpu'
    hyper['logger'] = 'local'
    bl = hyper["omega_0"] = [int(hyper["height"] / 6)]
    l = hyper["block_limits"][0]
    hyper["max_epochs_per_stage"] = 50
    for fname in ['data/div2k/test_data/07.png']:
        hyper['data_path'] = fname
        train_dataset, test_dataset = get_database(hyper, train_test=True)
        for a in [['sine', 'sine'], ["ffm", "relu"]]:
            project_name = hyper["project_name"] = "tuner_vs_ffm"
            # Define model
            model = HybridModel(
                 hyper['in_features'],
                    hyper['hidden_features'][0],
                    hyper['hidden_layers'],
                    hyper['out_features'],
                    hyper['omega_0'][0],
                    hyper['hidden_omega_0'],
                    activations=a,
                    period=hyper['period'],
                    low_range=l,
                    bandlimit = bl[0]
            )
            model.to(device)

            # Define bounds
            if a[0] == 'sine' == a[1]:
                bounds = hyper['bounds'][0]
                # init [1, 10, 4, 3, 12] would have bounds vector [c1, c2, c1, c1, c2]
                bounds = construct_bound(model, hyper["bounds"][0], hyper['block_limits'][0],
                                         hyper['period']) / hyper['hidden_omega_0']

            model.to(device)

            optimizer = torch.optim.Adam(lr=hyper['lr'],
                                            params=model.parameters())
            last_epoch_loss = 1000000
            total_epochs_trained = 0
            epochs = hyper['max_epochs_per_stage']
            dataset = train_dataset[0]
            loss_list = []
            for epoch in range(epochs):
                for batch in dataset:

                    optimizer.zero_grad()
                    model.zero_grad()

                    X, gt_dict = batch['c0']
                    pred = model(X['coords'].to(device))
                    out_dict = {'model_out': {'output': pred[0]}}

                    loss_dict = mse_loss(out_dict, gt_dict, device=device)
                    loss = loss_dict['d0'] * 1.0

                    loss.backward()
                    optimizer.step()

                    if a[0] == 'sine' == a[1]:
                        for name, weight in model.state_dict().items():
                            if '1' in name and 'weight' in name:
                                with torch.no_grad():
                                    torch.clamp_(weight,
                                                 min=-bounds,
                                                 max=bounds)
                    running_loss = {}
                    for key, value in loss_dict.items():
                        running_loss[key] = (running_loss.get(key, 0.0)
                                            + value.item())

                epoch_loss = {key: value / len(dataset)
                            for key, value in running_loss.items()}
                loss_list.append(epoch_loss['d0'])
                total_epoch_loss = sum(epoch_loss.values())
                total_epochs_trained += 1
                loss_gain = abs((total_epoch_loss - last_epoch_loss)
                                / total_epoch_loss)
                last_epoch_loss = total_epoch_loss

            name = project_name + "_" + os.path.basename(hyper["data_path"]).split('.')[0]
            dif = "tuner" if a[0] == 'sine' else "ffm"
            hf = ''.join(map(str, hyper['hidden_features'][0]))
            b = '-'.join(map(str, hyper['bounds'][0]))

            tname = f"{dif}_{name}_hf{hf}_b{bl[0]}_l{l}_Ep{epochs}_bo{b}"
            metrics = log_data(model,
                               train_dataset[0],
                               test_dataset[0],
                               hyper,
                               tname,
                               loss=loss_list)
