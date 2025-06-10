import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from taming.training.handler_base import log_data
from networks.siren import Siren
from utils import load_hyperparameters, get_database, mse_loss, construct_bound


if __name__ == "__main__":
    torch.manual_seed(777)

    # -- hyperparameters in configs --#
    hyper = load_hyperparameters("configs/config_init.yml")
    project_name = hyper["project_name"] = "deep_bounds"
    device = 'cuda:0' if hyper['device'] == 'cuda' else 'cpu'
    l = hyper["block_limits"] = [10]
    hw1 = []
    hw2 = []
    m1 = []
    m2 = []
    for img in os.listdir("data/kodak"):
        hyper["data_path"] = f"data/kodak/{img}"
        train_dataset, test_dataset = get_database(hyper, True)
        for init in ["tuner"]:
            # b = hyper["omega_0"] = [30] if init == "siren" else [int(hyper["height"] / 6)]
            b = hyper["omega_0"] = [int(hyper["height"] / 4)]
            if "macaws" not in img: continue
            p = hyper["period"] = 0 if init == "siren" else 3
            mode = "uniform" if init == "random" else "sampling"

            model = Siren(in_features=hyper['in_features'],
                        hidden_features=hyper['hidden_features'][0],
                        hidden_layers=hyper['hidden_layers'],
                        out_features=hyper['out_features'],
                        first_omega_0=hyper['omega_0'][0],
                        hidden_omega_0=hyper['hidden_omega_0'],
                        period=hyper["period"],
                        mode=mode,
                        low_range=hyper["block_limits"][0],
                        bandlimit = b[0])
            model.to(device)
            if init == "tuner":
                b1 = construct_bound(model, hyper["bounds"][0], l[0], p) / hyper['hidden_omega_0']
                b2 = hyper["bounds"][1] / hyper['hidden_omega_0']
            optimizer = torch.optim.Adam(lr=hyper['lr'],
                                        params=model.parameters())
            last_epoch_loss = 1000000
            total_epochs_trained = 0
            current_loss_tol = 1e-5
            current_diff_tol = 1e-5
            epochs = hyper['max_epochs_per_stage']
            dataset = train_dataset[0]
            loss_log = []
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
                    if init == "tuner":
                        with torch.no_grad():
                            for name, weight in model.state_dict().items():
                                if '0' not in name and 'weight' in name:
                                    bound = b1 if '1' in name else b2
                                    torch.clamp_(weight, min=-bound, max=bound)
                        hw2.append(model.net[2].linear.weight.abs().max().cpu().detach()*hyper["hidden_omega_0"])
                        hw1.append(model.net[1].linear.weight.abs().max().cpu().detach()*hyper["hidden_omega_0"])
                        m2.append(model.net[2].linear.weight.abs().mean().cpu().detach()*hyper["hidden_omega_0"])
                        m1.append(model.net[1].linear.weight.abs().mean().cpu().detach()*hyper["hidden_omega_0"])
                        
                    running_loss = {}
                    for key, value in loss_dict.items():
                        running_loss[key] = (running_loss.get(key, 0.0)
                                            + value.item())

                epoch_loss = {key: value / len(dataset)
                            for key, value in running_loss.items()}
                # if (epoch + 1) % 100 == 0: print(f"Epoch: {epoch + 1}  -  Loss:", epoch_loss['d0'])
                total_epoch_loss = sum(epoch_loss.values())
                total_epochs_trained += 1
                loss_gain = abs((total_epoch_loss - last_epoch_loss)
                                / total_epoch_loss)
                last_epoch_loss = total_epoch_loss
                loss_log.append(total_epoch_loss)

            name = project_name + "_" + os.path.basename(hyper["data_path"]).split('.')[0]
            dif = 'S' if init == "siren" else 'T' if init == "tuner" else 'R'
            hf = ''.join(map(str, hyper['hidden_features'][0]))
            tname = f"{dif}_{name}_hf{hf}_b{b[0]}_l{l[0]}_Ep{epochs}_P{p}"
            print(model.net[1].linear.weight.abs().max()*hyper["hidden_omega_0"])
            print(model.net[2].linear.weight.abs().max()*hyper["hidden_omega_0"])
            metrics = log_data(model,
                               train_dataset[0],
                               test_dataset[0],
                               hyper,
                               tname,
                               loss=loss_log)
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ep_to_start = 10
            lenght = len(hw1[ep_to_start:])
            ax[0].plot(np.linspace(ep_to_start, 200, lenght), np.array(hw1)[ep_to_start:], label='Hidden Layer 1')
            ax[0].plot(np.linspace(ep_to_start, 200, lenght), np.array(m1)[ep_to_start:], label='Mean')
            ax[0].plot(np.linspace(ep_to_start, 200, lenght), np.ones_like(hw1[ep_to_start:])*hyper["bounds"][0][0], label='c_L')
            ax[0].plot(np.linspace(ep_to_start, 200, lenght), np.ones_like(hw1[ep_to_start:])*hyper["bounds"][0][1], label='c_H')
            ax[0].legend()
            ax[1].plot(np.linspace(ep_to_start, 200, lenght), np.array(hw2)[ep_to_start:], label='Hidden Layer 2')
            ax[1].plot(np.linspace(ep_to_start, 200, lenght), np.array(m2)[ep_to_start:], label='Mean')
            ax[1].plot(np.linspace(ep_to_start, 200, lenght), np.ones_like(hw2[ep_to_start:])*hyper["bounds"][1], label='bound')
            ax[1].legend()
            ax[2].plot(np.arange(ep_to_start, 200), loss_log[ep_to_start:], label='Loss')
            plt.show()
            plt.close()
