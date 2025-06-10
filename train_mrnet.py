import csv
import os
from pathlib import Path
import numpy as np
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.training.listener import TrainingListener

from networks.mrnet import Initializer
from training.optimizers import ClampOptimizationHandler
from utils import (create_clamps, load_hyperparameters,
                   get_database)


RANDOM_SEED = 777
def optim_handler(model, optimizer, loss_function, loss_weights):
    return ClampOptimizationHandler(
        model, optimizer, loss_function, loss_weights,
        bounds)


if __name__ == "__main__":

    torch.manual_seed(RANDOM_SEED)

    hyper = load_hyperparameters("configs/config_init.yml")
    project_name = hyper["project_name"] = "train_mrnet"
    hyper["bounds"] = [1.0, 0.4]
    hyper['max_epochs_per_stage'] = 100
    init_freqs = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156]
        ]
    initializer = Initializer(hyper, init_freqs=init_freqs,
                                bias_init=False, init_W=True)
    mrmodel = initializer.get_model()
    freqs = initializer.input_freqs

    bounds = torch.tensor(hyper["bounds"])
    logger = TrainingListener(
        project_name,
        (f"{os.path.basename(hyper['data_path'])}_p{hyper['period']}_cl" +
            f"{'-'.join(map(str, hyper['bounds']))}_lims_" + 
            '-'.join(map(str, hyper['block_limits']))),
        hyper,
        Path(hyper.get("log_path", "runs")),
    )
    train_dataset, test_dataset = get_database(hyper, train_test=True)
    mrtrainer = MRTrainer.init_from_dict(
        mrmodel, train_dataset, test_dataset, logger, hyper,
        optim_handler=optim_handler
    )
    mrtrainer.train(hyper["device"])
