import csv
import os
import torch
import torch.nn.functional as F

from logs.handler_base import log_data
from networks.siren import Siren
from utils import load_hyperparameters, get_database


def mse_loss(output_dict, gt_dict, **kwargs):
    pred: torch.Tensor = output_dict['model_out']
    pred = pred['output']
    gt = gt_dict['d0'].to(device)
    loss_dict = {'d0': F.mse_loss(pred, gt, device)}
    return loss_dict


class OptimizationHandler:
    def __init__(self, model, optimizer, loss_function) -> None:
        self.model = model
        # TODO: should it be constructed here?
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_weights = 1.0

    def _pre_process(self):
        self.optimizer.zero_grad()
        # not all model's parameters are updated by the optimizer
        self.model.zero_grad()

    def _post_process(self, loss_dict):
        loss = loss_dict['d0'] * self.loss_weights

        loss.backward()
        self.optimizer.step()

        running_loss = {}
        for key, value in loss_dict.items():
            running_loss[key] = (running_loss.get(key, 0.0)
                                 + value.item())
        return running_loss

    def optimize(self, batch, device):
        """This function should be overwritten for custom losses"""
        # why c0?
        X, gt_dict = batch['c0']
        X['coords'] = X['coords'].to(device)
        out_dict = self.model(X)

        loss_dict = self.loss_function(out_dict, gt_dict, device=device)
        print(loss_dict['d0'])
        return loss_dict

    def __call__(self, batch, device):
        self._pre_process()
        loss_dict = self.optimize(batch, device)
        return self._post_process(loss_dict)


if __name__ == "__main__":
    torch.manual_seed(777)

    for test_set in [True, False]:
        psnr = {}
        psnr_grad = {}

        # -- hyperparameters in configs --#
        hyper = load_hyperparameters(
            "/home/diana/taming/taming/configs/config_init.yml")
        project_name = hyper["project_name"] = "bacon"
        train_dataset, test_dataset = get_database(hyper)
        device = 'cuda:0'
        f = hyper['omega_0'][0]
        hf = hyper['hidden_features'][0][0]
        m2 = Siren(in_features=2,
                   hidden_features=[24, 48, 96],
                   hidden_layers=2,
                   out_features=3,
                   first_omega_0=60,
                   hidden_omega_0=60)
        # m2 = Siren(in_features=hyper['in_features'],
        #            hidden_features=hyper['hidden_features'][0],
        #            hidden_layers=hyper['hidden_layers'],
        #            out_features=hyper['out_features'],
        #            first_omega_0=hyper['omega_0'],
        #            hidden_omega_0=hyper['hidden_omega_0'])
        m2.to(device)
        optimizer = torch.optim.Adam(lr=0.005,
                                     params=m2.parameters())
        optim_handler = OptimizationHandler(m2, optimizer, mse_loss)
        last_epoch_loss = 1000000
        total_epochs_trained = 0
        current_loss_tol = 1e-5
        current_diff_tol = 1e-5
        epochs = hyper['max_epochs_per_stage']
        dataset = train_dataset[0]
        for epoch in range(epochs):
            for batch in dataset:
                running_loss = optim_handler(batch, device)

            epoch_loss = {key: value / len(dataset)
                          for key, value in running_loss.items()}
            total_epoch_loss = sum(epoch_loss.values())
            total_epochs_trained += 1
            loss_gain = abs((total_epoch_loss - last_epoch_loss)
                            / total_epoch_loss)
            last_epoch_loss = total_epoch_loss

        name = os.path.basename(hyper["data_path"])
        dif = 'Te' if test_set else 'Tr'
        tname = f"{name}_{dif}_hf{hf}_b{f}_Ep{epochs}"
        metrics = log_data(
            m2, train_dataset[0], test_dataset[0], device,
            hyper['batch_size'], hyper['channels'], tname
            )
        psnr[name] = metrics[0]
        psnr_grad[name] = metrics[1]
    directory = '/home/diana/taming/taming/results/compare_bacon/'
    with open(directory + dif + '_psnr.csv', 'w') as output:
        writer = csv.writer(output)
        for key, value in psnr.items():
            writer.writerow([key, value, psnr_grad[key]])
