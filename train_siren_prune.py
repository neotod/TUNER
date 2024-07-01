import os
import torch
import torch.nn.functional as F

from logs.handler_base import log_data
from networks.siren import Siren
from utils import load_hyperparameters, get_database
from training.pruner import Pruner


def mse_loss(output_dict, gt_dict, **kwargs):
    pred: torch.Tensor = output_dict['model_out']
    pred = pred['output']
    gt = gt_dict['d0'].to(device)
    loss_dict = {'d0': F.mse_loss(pred, gt, device)}
    return loss_dict


if __name__ == "__main__":
    torch.manual_seed(777)

    psnr = {}
    psnr_grad = {}

    # -- hyperparameters in configs --#
    hyper = load_hyperparameters(
        "/home/diana/taming/taming/configs/config_init.yml")
    project_name = hyper["project_name"] = "bacon"
    train_dataset, test_dataset = get_database(hyper)
    device = 'cuda:0' if hyper['device'] == 'cuda' else 'cpu'
    hf = hyper['hidden_features'][0][0]
    low_range = 10
    model = Siren(in_features=hyper['in_features'],
                  hidden_features=hyper['hidden_features'][0],
                  hidden_layers=hyper['hidden_layers'],
                  out_features=hyper['out_features'],
                  first_omega_0=hyper['omega_0'][0],
                  hidden_omega_0=hyper['hidden_omega_0'],
                  period=2, mode="sampling", low_range=low_range)
    model.to(device)

    optimizer = torch.optim.Adam(lr=0.0001,
                                 params=model.parameters())
    last_epoch_loss = 1000000
    total_epochs_trained = 0
    current_loss_tol = 1e-5
    current_diff_tol = 1e-5
    epochs = hyper['max_epochs_per_stage']
    dataset = train_dataset[0]

    pruner = Pruner(model, p=1, dim=1)

    for epoch in range(epochs):
        if epoch == 50:
            pruner.prune([0.5, 0.5])

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

            running_loss = {}
            for key, value in loss_dict.items():
                running_loss[key] = (running_loss.get(key, 0.0)
                                     + value.item())

        epoch_loss = {key: value / len(dataset)
                      for key, value in running_loss.items()}
        print(epoch_loss)
        total_epoch_loss = sum(epoch_loss.values())
        total_epochs_trained += 1
        loss_gain = abs((total_epoch_loss - last_epoch_loss)
                        / total_epoch_loss)
        last_epoch_loss = total_epoch_loss

    name = os.path.basename(hyper["data_path"]).split('.')[0]
    dif = 'Te'
    bandlimit = hyper['omega_0'][0]
    tname = f"{name}_{dif}_hf{hf}_b{bandlimit}_l{low_range}_Ep{epochs}"
    metrics = log_data(
        model, train_dataset[0], test_dataset[0], device,
        hyper['batch_size'], hyper['channels'], tname
        )
    print(list(model.net[0].named_buffers()))