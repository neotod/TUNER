import os
from pathlib import Path
import torch

# from mrnet.training.listener import TrainingListener
from training.tracking_logger import TrackTrainingListener

from training.trainer import MRTrainer
from networks.mrnet import MRFactory
from utils import (load_hyperparameters,
                   get_database, get_optim_handler)


if __name__ == "__main__":
    torch.manual_seed(777)

    # -- hyperparameters in configs --#
    # hyper = load_hyperparameters(
    #     "/home/diana/taming/taming/configs/config_init.yml")
    hyper = load_hyperparameters("./configs/config_init.yml")
    project_name = hyper["project_name"] = "deep_networks"
    train_dataset, test_dataset = get_database(hyper, False, 0.1)
    m2 = MRFactory.from_dict(hyper)
    optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
    name = os.path.basename(hyper["data_path"])

    logger = TrackTrainingListener(
        project_name,
        f"{name[0:7]}{hyper['color_space'][0]}",
        hyper,
        Path(hyper.get("log_path", "runs")))
    mrtrainer = MRTrainer.init_from_dict(m2, train_dataset,
                                         test_dataset, logger, hyper,
                                         optim_handler=optim_handler)
    mrtrainer.train(hyper["device"])
