import csv
import os
from pathlib import Path
import numpy as np
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.training.listener import TrainingListener

from networks.initializer import Initializer
# from training.tracking_logger import TrackTrainingListener
from training.optimizers import ClampOptimizationHandler
# from mrnet.training.optimizer import OptimizationHandler
from utils import (create_clamps, load_hyperparameters,
                   get_database)


RANDOM_SEED = 777

if __name__ == "__main__":

    torch.manual_seed(RANDOM_SEED)

    psnr_bounds = []
    psnr_grad_bounds = []
    configs = []
    image_paths = ['data/kodak/' + fname for fname in os.listdir('./data/kodak')]

    for fname in image_paths:
        for hidden_features in [[[416, 416]], [[416, 294, 294]]]:
            hyper = load_hyperparameters("configs/config_init.yml")
            project_name = hyper["project_name"] = "1hl_VS_2hl"
            hyper['data_path'] = fname
            hyper['hidden_features'] = hidden_features
            n_hl = len(hidden_features[0]) - 1
            hyper['hidden_layers'] = n_hl
            hyper['width'] = 1024
            hyper['height'] = 1024
            hyper['max_epochs_per_stage'] = 100
            bounds = [[1.2, 0.4], 0.5]
            init_freqs = [[0, 1, 2, 3, 4, 5, 6, 7],
                          [0, 13, 26, 39, 52, 65, 78, 91,
                           104, 117, 130, 143, 156]]
                        # [int(float(i) * float(90) / 7) for i in range(1, 8)]]
            initializer = Initializer(hyper, init_freqs=init_freqs,
                                      bias_init=False, init_W=True)
            mrmodel = initializer.get_model()
            first_weight = mrmodel.stages[0].first_layer.linear.weight
            freqs = first_weight * hyper['period'] / (2 * torch.pi)

            # [1, 10, 4, 3, 12] -> [c1, c2, c1, c1, c2]
            b = '-'.join(map(str, bounds))

            bounds[0] = create_clamps(bounds[0],
                                      hyper['block_limits'],
                                      freqs)
            bounds = bounds[:n_hl]

            def optim_handler(model, optimizer, loss_function, loss_weights):
                return ClampOptimizationHandler(
                    model, optimizer, loss_function, loss_weights,
                    bounds)

            print("Model: ", type(mrmodel))
            name = os.path.basename(hyper["data_path"])
            logger = TrainingListener(
                project_name,
                (f"{name[0:7]}" +
                    f"p{hyper['period']}_cl{b}" +
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

    #         savedir = logger.handler.logger.savedir
    #         file = os.path.join(savedir, 'psnr.csv')
    #         with open(file) as csv_file:
    #             csv_reader = csv.reader(csv_file, delimiter=',')
    #             for line in csv_reader:
    #                 psnr_bounds.append(float(line[1]))
    #         file2 = os.path.join(savedir, 'grad_psnr.csv')
    #         with open(file2) as csv_file2:
    #             csv_reader2 = csv.reader(csv_file2, delimiter=',')
    #             for i, line2 in enumerate(csv_reader2):
    #                 if i == 0:
    #                     continue
    #                 psnr_grad_bounds.append(float(line2[0]))

    # print('PSNR:', psnr_bounds)
    # print('Grad PSNR:', psnr_grad_bounds)
    # array = np.array([psnr_bounds])
    # grad_psnrs = np.array([psnr_bounds])
    # name = './results/bounds/psnr.csv'
    # name2 = './results/bounds/grad_psnr_grad.csv'
    # np.savetxt(name2, grad_psnrs, delimiter="\n")
    # np.savetxt(name, array, delimiter="\n")
