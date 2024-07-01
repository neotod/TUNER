import os
from pathlib import Path
import torch

from mrnet.training.listener import TrainingListener
from mrnet.training.optimizer import OptimizationHandler

from training.trainer import MRTrainer
from networks.mrnet import MRFactory
from utils import load_hyperparameters, get_database


class BoundOptimizationHandler(OptimizationHandler):
    def _post_process(self, loss_dict):
        b = self.model.stages[0].middle_layers[0].bounds.data = \
                torch.abs(self.model.stages[0].first_layer.bounds)

        loss_dict['bound'] = torch.norm(b, p=1) / len(b)
        running_loss = super()._post_process(loss_dict)
        return running_loss


if __name__ == "__main__":
    torch.manual_seed(777)

    # -- hyperparameters in configs --#
    hyper = load_hyperparameters(
        "/home/diana/taming/taming/configs/config_init.yml")
    # hyper = load_hyperparameters("./configs/config_init.yml")
    project_name = hyper["project_name"] = "learn_bounds"
    train_dataset, test_dataset = get_database(hyper, False, 0.1)
    m2 = MRFactory.from_dict(hyper)
    optim_handler = BoundOptimizationHandler
    name = os.path.basename(hyper["data_path"])

    logger = TrainingListener(project_name,
                              f"{name[0:7]}{hyper['color_space'][0]}",
                              hyper,
                              Path(hyper.get("log_path", "runs")))
    mrtrainer = MRTrainer.init_from_dict(m2, train_dataset,
                                         test_dataset, logger, hyper,
                                         optim_handler=optim_handler)
    print(m2.stages[0].middle_layers[0].bounds)
    mrtrainer.train(hyper["device"])
    print(m2.stages[0].middle_layers[0].bounds)
