import os
import torch
import datetime

from taming.training.handler_base import log_data
from networks.siren import Siren
from utils import load_hyperparameters, get_database, mse_loss, construct_bound


if __name__ == "__main__":
    torch.manual_seed(777)

    # -- hyperparameters in configs --#
    hyper = load_hyperparameters("configs/config_init.yml")
    project_name = hyper["project_name"] = "compare_banf"
    device = 'cuda:0' if hyper['device'] == 'cuda' else 'cpu'
    l = hyper["block_limits"] = [10]
    for imgfolder in os.listdir("data/data_triangle"):
        hyper["data_path"] = f"data/data_triangle/{imgfolder}/gt_256.png"
        train_dataset, test_dataset = get_database(hyper, False, width_test=512, height_test=512)
        for res in ["64", "128", " 256"]: # 
            # Match size and epochs to each stage of BANF net size and training time
            hyper["max_epochs_per_stage"] = 4500 if res == "64" else 7500 if res == "128" else 6000
            hyper["hidden_features"] = [126] if res == "64" else [223] if res == "128" else [416]
            
            b = [(int(res) // 2) // 3]

            model = Siren(in_features=hyper['in_features'],
                        hidden_features=hyper['hidden_features'][0],
                        hidden_layers=hyper['hidden_layers'],
                        out_features=hyper['out_features'],
                        first_omega_0=hyper['omega_0'][0],
                        hidden_omega_0=hyper['hidden_omega_0'],
                        period=hyper["period"],
                        low_range=hyper["block_limits"][0],
                        bandlimit = b[0],
                        perc_low_freqs=0.8,)
            model.to(device)
            bound = construct_bound(model,
                                 hyper["bounds"][0],
                                 l[0],
                                 hyper["period"]) / hyper['hidden_omega_0']
            optimizer = torch.optim.Adam(lr=hyper['lr'],
                                        params=model.parameters())
            last_epoch_loss = 1000000
            total_epochs_trained = 0
            current_loss_tol = 1e-5
            current_diff_tol = 1e-5
            epochs = hyper['max_epochs_per_stage']
            dataset = train_dataset[0]
            loss_log = []
            loop_start = datetime.datetime.now()
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
                    with torch.no_grad():
                        for name, weight in model.state_dict().items():
                            if '0' not in name and 'weight' in name:
                                torch.clamp_(weight, min=-bound, max=bound)
                        
                    running_loss = {}
                    for key, value in loss_dict.items():
                        running_loss[key] = (running_loss.get(key, 0.0)
                                            + value.item())

                epoch_loss = {key: value / len(dataset)
                            for key, value in running_loss.items()}
                total_epoch_loss = sum(epoch_loss.values())
                total_epochs_trained += 1
                loss_gain = abs((total_epoch_loss - last_epoch_loss)
                                / total_epoch_loss)
                last_epoch_loss = total_epoch_loss
                loss_log.append(total_epoch_loss)
            print(f"Training done in: {datetime.datetime.now() - loop_start}")
            name = project_name + "_" + str(hyper["data_path"].split('/')[2])
            hf = hyper['hidden_features'][0]
            tname = f"{res}_{name}_hf{hf}_b{b[0]}_l{l[0]}_Ep{epochs}"
            metrics = log_data(model,
                               train_dataset[0],
                               test_dataset[0],
                               hyper,
                               tname,
                               loss=loss_log,
                               bandlimit_metric=int(res))