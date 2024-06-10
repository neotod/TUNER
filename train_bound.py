import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.training.listener import TrainingListener

from networks.mrnet import MRFactory
from training.optimizers import ClampOptimizationHandler
from utils import (create_clamps, load_hyperparameters,
                   get_database)


RANDOM_SEED = 777

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    hyper = load_hyperparameters("/home/diana/taming/taming/configs/config_init.yml")

    project_name = hyper["project_name"] = "bound"
    mrmodel = MRFactory.from_dict(hyper)
    freqs = mrmodel.stages[0].first_layer.linear.weight * hyper['period'] / (2 * torch.pi)
    clamps = create_clamps(hyper['clamps'], hyper['block_limits'],
                           freqs)

    def optim_handler(model, optimizer, loss_function, loss_weights):
        return ClampOptimizationHandler(
            model, optimizer, loss_function, loss_weights,
            clamps)
    print("Model: ", type(mrmodel))
    name = os.path.basename(hyper["data_path"])
    logger = TrainingListener(
        project_name,
        (f"{name[0:7]}{hyper['color_space'][0]}" +
            f"p{hyper['period']}_cl{'-'.join(map(str, hyper['clamps']))}" +
            f"lims{'-'.join(map(str, hyper['block_limits']))}"),
        hyper,
        Path(hyper.get("log_path", "runs")),
    )
    train_dataset, test_dataset = get_database(hyper, train_test=True)
    mrtrainer = MRTrainer.init_from_dict(
        mrmodel, train_dataset, test_dataset, logger, hyper,
        optim_handler=optim_handler
    )
    mrtrainer.train(hyper["device"])
