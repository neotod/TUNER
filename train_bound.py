import csv
import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
# from mrnet.training.listener import TrainingListener

from networks.mrnet import MRFactory
from networks.initializer import Initializer
from training.tracking_logger import TrackTrainingListener
from training.optimizers import ClampOptimizationHandler
from mrnet.training.optimizer import OptimizationHandler
from utils import (create_clamps, load_hyperparameters,
                   get_database)


RANDOM_SEED = 777

if __name__ == "__main__":

    torch.manual_seed(RANDOM_SEED)

    psnr_bounds = []
    psnr_grad_bounds = []

    for period in [0, 2.1]:
        hyper = load_hyperparameters("configs/config_init.yml")
        hyper['period'] = period
        project_name = hyper["project_name"] = "bound"

        if hyper['period']:
            init_freqs = [[0, 1, 2, 3, 4, 5, 6, 7],
                        #   [0, 13, 26, 39, 52, 65, 78, 91,
                        #    104, 117, 130, 143, 156]]
                          [16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82]]
            initializer = Initializer(hyper, init_freqs=init_freqs,
                                      bias_init=True, init_W=False)
            # initializer.initialize_middle_layer()
            mrmodel = initializer.get_model()


            first_weight = mrmodel.stages[0].first_layer.linear.weight
            freqs = first_weight * hyper['period'] / (2 * torch.pi)

            # [1, 10, 4, 3, 12] -> [c1, c2, c1, c1, c2]
            bounds_1hl = create_clamps(hyper['bounds'][0],
                                       hyper['block_limits'],
                                       freqs)
            bounds = [bounds_1hl, hyper['bounds'][1]]

            def optim_handler(model, optimizer, loss_function, loss_weights):
                return ClampOptimizationHandler(
                    model, optimizer, loss_function, loss_weights,
                    bounds)
        else:
            mrmodel = MRFactory.from_dict(hyper)
            optim_handler = OptimizationHandler
        print("Model: ", type(mrmodel))
        name = os.path.basename(hyper["data_path"])
        logger = TrackTrainingListener(
            project_name,
            (f"{name[0:7]}" +
                f"p{hyper['period']}_cl{'-'.join(map(str, hyper['bounds']))}" +
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

        savedir = logger.handler.logger.savedir
        file = os.path.join(savedir, 'psnr.csv')
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                psnr_bounds.append(float(line[1]))
        file2 = os.path.join(savedir, 'grad_psnr.csv')
        with open(file2) as csv_file2:
            csv_reader2 = csv.reader(csv_file2, delimiter=',')
            for i, line2 in enumerate(csv_reader2):
                if i == 0:
                    continue
                psnr_grad_bounds.append(float(line2[0]))

    print('PSNR:', psnr_bounds)
    print('Grad PSNR:', psnr_grad_bounds)
