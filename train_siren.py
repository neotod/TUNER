import os
from pathlib import Path
import torch
from mrnet.training.trainer import MRTrainer
# from mrnet.networks.mrnet import MRFactory
from mrnet.training.listener import TrainingListener

from networks.siren import Siren
from utils import (load_hyperparameters,
                   get_database, get_optim_handler)

if __name__ == "__main__":
    torch.manual_seed(777)

    # -- hyperparameters in configs --#
    hyper = load_hyperparameters("../configs/config_init.yml")
    project_name = hyper["project_name"] = "learn_bounds"
    train_dataset, test_dataset = get_database(hyper, False, 0.1)

    hyper['omega_0'] = 64
    hyper['period'] = 0
    m2 = Siren(in_features=2, hidden_features=10, hidden_layers=2,
               out_features=3, first_omega_0=60, hidden_omega_0=60)
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
    name = os.path.basename(hyper["data_path"])

    logger = TrainingListener(project_name,
                              f"{name[0:7]}{hyper['color_space'][0]}",
                              hyper,
                              Path(hyper.get("log_path", "runs")))
    mrtrainer = MRTrainer.init_from_dict(m2, train_dataset,
                                         test_dataset, logger, hyper,
                                         optim_handler=optim_handler)
    mrtrainer.train(hyper["device"])

