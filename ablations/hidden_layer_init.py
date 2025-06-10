import csv
import os
from pathlib import Path
import numpy as np
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.training.listener import TrainingListener

from networks.mrnet import Initializer
from mrnet.training.optimizer import OptimizationHandler
from utils import (load_hyperparameters,
                   get_database)


RANDOM_SEED = 777

if __name__ == "__main__":

    torch.manual_seed(RANDOM_SEED)

    image_paths = ['data/kodak/' + fname for fname in os.listdir('./data/kodak')]
    for data_type in ['train', 'test']:
        for W_init_type in ['uniform', 'ours']:
            psnr_bounds = []
            psnr_grad_bounds = []
            for fname in image_paths:
                hyper = load_hyperparameters("configs/config_init.yml")
                project_name = hyper["project_name"] = "W_init_ablation"
                hyper['data_path'] = fname
                hyper['hidden_layers'] = 1
                hyper['hidden_features'] = [[416, 1024]]
                hyper['hidden_omega_0'] = 85
                hyper['max_epochs_per_stage'] = 10
                hyper['batch_size'] = 256 * 256
                hyper['clamps'] = [1.0, .2]
                hyper['width'] = 1024
                hyper['height'] = 1024
                init_W = True if W_init_type == 'ours' else False
                init_freqs = [[0, 1, 2, 3, 4, 5, 6, 7],
                              [13, 26, 39, 52, 65, 78, 91, 104,
                               117, 130, 143, 156]]
                initializer = Initializer(hyper, init_freqs=init_freqs,
                                          bias_init=True, init_W=init_W)
                mrmodel = initializer.get_model()
                first_weight = mrmodel.stages[0].first_layer.linear.weight
                freqs = first_weight * hyper['period'] / (2 * torch.pi)
                optim_handler = OptimizationHandler

                print("Model: ", type(mrmodel))
                name = os.path.basename(hyper["data_path"])
                logger = TrainingListener(
                    project_name,
                    (f"{name[0:7]}_{data_type}_{W_init_type}"),
                    hyper,
                    Path(hyper.get("log_path", "runs")),
                )
                train_test = False if data_type == 'train' else True
                train_dataset, test_dataset = get_database(
                    hyper, train_test=train_test
                    )
                mrtrainer = MRTrainer.init_from_dict(
                    mrmodel, train_dataset, test_dataset, logger, hyper,
                    optim_handler=optim_handler
                )
                mrtrainer.train(hyper["device"])
