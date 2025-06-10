import os
import torch
import torch.nn.functional as F

from taming.training.handler_base import log_data
from networks.siren import Siren
from utils import load_hyperparameters, get_database, mse_loss, construct_bound, get_forward


if __name__ == "__main__":
    torch.manual_seed(777)

    hyper = load_hyperparameters("configs/config_init.yml")
    project_name = hyper["project_name"] = "deep_net"
    device = 'cuda:0' if hyper['device'] == 'cuda' else 'cpu'
    b = [int(hyper["height"] / 6)]
    l = hyper["block_limits"] = [int(b[0] / 3)]
    for img in os.listdir("data/kodak"):
        hyper["data_path"] = f"data/kodak/{img}"
        train_dataset, test_dataset = get_database(hyper, True)
        for init in ["_", "tuner"]:  
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
                        bandlimit = b[0],
                        perc_low_freqs=0.7,)
            model.to(device)
            if init == "tuner":
                b1 = construct_bound(model, hyper["bounds"][0], l[0], p)
                b2 = (hyper["bounds"][1]) * torch.ones_like(model.net[2].linear.weight)
                with torch.no_grad():
                    model.net[1].bound = torch.nn.Parameter(b1)
                    model.net[2].bound = torch.nn.Parameter(b2)
                    model.net[1].forward = get_forward(model, hyper, 1)
                    model.net[2].forward = get_forward(model, hyper, 2)

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
                    if init == "tuner":
                        bh1 = model.net[1].bound.data = torch.abs(model.net[1].bound)
                        bh2 = model.net[2].bound.data = torch.abs(model.net[2].bound)
                        loss_dict['bound'] = torch.norm(bh1, p=2) / len(bh1) + \
                            torch.norm(bh2, p=1) / len(bh2)

                    loss = sum([loss_dict[key] * hyper["loss_weights"].get(key, 1.0)
                                for key in loss_dict.keys()])
                    
                    loss.backward()
                    optimizer.step()                        
                    running_loss = {}
                    for key, value in loss_dict.items():
                        running_loss[key] = (running_loss.get(key, 0.0)
                                            + value.item())

                if (epoch + 1) % 500 == 0: print(f"Epoch: {epoch + 1}  -  Loss:", running_loss["d0"].item())
                epoch_loss = {key: value / len(dataset)
                            for key, value in running_loss.items()}
                total_epoch_loss = sum(epoch_loss.values())
                total_epochs_trained += 1
                loss_gain = abs((total_epoch_loss - last_epoch_loss)
                                / total_epoch_loss)
                last_epoch_loss = total_epoch_loss
                loss_log.append(total_epoch_loss)
            name = project_name + "_" + os.path.basename(hyper["data_path"]).split('.')[0]
            dif = 'S' if init == "siren" else 'T' if init == "tuner" else 'R' if "random" else "I"
            hf = ''.join(map(str, hyper['hidden_features'][0]))
            tname = f"{dif}_{name}_hf{hf}_b{b[0]}_l{l[0]}_Ep{epochs}_P{p}"

            metrics = log_data(model,
                               train_dataset[0],
                               test_dataset[0],
                               hyper,
                               tname,
                               loss=loss_log)
            
            
