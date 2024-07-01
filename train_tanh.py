import csv
import os
from pathlib import Path
import numpy as np
import torch

from mrnet.training.trainer import MRTrainer
from networks.mrnet import MRFactory
from mrnet.training.listener import TrainingListener
from mrnet.training.optimizer import OptimizationHandler

from utils import (create_clamps, load_hyperparameters,
                   get_database, get_optim_handler)


if __name__ == "__main__":
    torch.manual_seed(777)
    psnrs_images = []
    for test_set in [True]:
        for fname in ['barn_and_pond.png']:  # os.listdir("./data/img/kodak/"):  #
        # -- hyperparameters in configs --#
            psnr_bounds = psnr_grad_bounds = []
            # hyper = load_hyperparameters("../docs/configs/config_init.yml")
            # project_name = hyper["project_name"] = "learn_bounds"
            # hyper['data_path'] = "./data/img/kodak/" + fname
            # train_dataset, test_dataset = get_database(hyper, test_set, 0.1)

            # hyper['omega_0'] = 512
            # hyper['period'] = 0
            # m2 = MRFactory.from_dict(hyper)
            # optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
            # name = os.path.basename(hyper["data_path"])

            # logger = TrackTrainingListener(project_name,
            #                         f"{name[0:7]}{hyper['color_space'][0]}",
            #                         hyper,
            #                         Path(hyper.get("log_path", "runs")))
            # mrtrainer = MRTrainer.init_from_dict(m2, train_dataset,
            #                                     test_dataset, logger, hyper,
            #                                     optim_handler=optim_handler)
            # mrtrainer.train(hyper["device"])
            # savedir = logger.handler.logger.savedir
            # file = os.path.join(savedir, 'psnr.csv')
            # with open(file) as csv_file:
            #     csv_reader = csv.reader(csv_file, delimiter=',')
            #     for line in csv_reader:
            #         psnr_bounds.append(float(line[1]))
            # file2 = os.path.join(savedir, 'grad_psnr.csv')
            # with open(file2) as csv_file2:
            #     csv_reader2 = csv.reader(csv_file2, delimiter=',')
            #     for i, line2 in enumerate(csv_reader2):
            #         if i == 0:
            #             continue
            #         psnr_grad_bounds.append(float(line2[0]))


            # -- hyperparameters in configs --#
            hyper = load_hyperparameters("../docs/configs/config_init.yml")
            project_name = hyper["project_name"] = "learn_bounds"
            hyper['data_path'] = "./data/img/kodak/" + fname
            train_dataset, test_dataset = get_database(hyper, False, 0.1)
            hf = hyper['hidden_features']
            # S Net --------------------------------------------------------- #
            hyper['omega_0'] = 256
            torch.cuda.empty_cache()
            init_freqs = [[0, 1, 2, 3, 4, 5, 6, 7],
                        #   [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]]
                          [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 256]]
                        #   [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 85]]
                        #   [14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 171]]
            l1, l2 = len(init_freqs[0]), len(init_freqs[1])
            hf[0][0] = 4 * (l1 + l2 - 2) + 2 * ((l1 - 1) ** 2 + (l2 - 1) ** 2 + 1)
            hyper['block_limits'] = [int(max(init_freqs[1]) / 3)]
            # bound = torch.rand(hf[0][0])
            learn_bounds = True
            hyper['hidden_features'] = hf
            initializer = Initializer(hyper, init_freqs=init_freqs,
                                      bias_init=True, init_W=test_set)
            # initializer.initialize_middle_layer_with_identity()
            m2 = initializer.get_model()
            bound = create_clamps(hyper['clamps'], hyper['block_limits'],
                                  m2.frequencies)

            with torch.no_grad():
                boost = 30
                m2.stages[0].middle_layers[0].bound = \
                    torch.nn.Parameter(bound / boost)

            if learn_bounds:
                def forward(input):
                    W = (torch.tanh(hyper['hidden_omega_0'] *
                                    m2.stages[0].middle_layers[0].linear.weight) *
                        boost * m2.stages[0].middle_layers[0].bound)
                    x = (input @ W.T + hyper['hidden_omega_0'] *
                        m2.stages[0].middle_layers[0].linear.bias)  # UNDO
                    # print(m2.stages[0].middle_layers[0].bound * boost)
                    return torch.sin(x)
                m2.stages[0].middle_layers[0].forward = forward

                class BoundOptimizationHandler(OptimizationHandler):
                    def __init__(self, model, optimizer, loss_fn, weight_decay):
                        super().__init__(model, optimizer, loss_fn, weight_decay)
                        self.lam = torch.tensor(1 / (len(bound) * bound),
                                                requires_grad=False,
                                                device='cuda:0')

                    def _post_process(self, loss_dict):
                        # with torch.no_grad():
                        b = self.model.stages[0].middle_layers[0].bound.data = \
                                torch.abs(self.model.stages[0].middle_layers[0].bound)

                        # loss_dict['bound'] = b @ self.lam
                        loss_dict['bound'] = torch.norm(b, p=1) / len(b)
                        running_loss = super()._post_process(loss_dict)
                        return running_loss
                optim_handler = BoundOptimizationHandler
            else:
                m2.stages[0].middle_layers[0].bound.requires_grad = False
                optim_handler = (lambda m, o, f, w:
                                ClampOptimizationHandler(m, o, f, w,
                                                        bound=bound))
            diff = 'T' if test_set else 'F'
            name = diff + os.path.basename(hyper["data_path"])

            logger = TrackTrainingListener(
                project_name, f"{name[0:7]}{hyper['color_space'][0]}",
                hyper, Path(hyper.get("log_path", "runs")))
            mrtrainer = MRTrainer.init_from_dict(m2, train_dataset,
                                                test_dataset, logger, hyper,
                                                optim_handler=optim_handler)
            mrtrainer.train(hyper["device"])
            # savedir = logger.handler.logger.savedir
            # file = os.path.join(savedir, 'psnr.csv')
            # with open(file) as csv_file:
            #     csv_reader = csv.reader(csv_file, delimiter=',')
            #     for line in csv_reader:
            #         psnr_bounds.append(float(line[1]))
            # file2 = os.path.join(savedir, 'grad_psnr.csv')
            # with open(file2) as csv_file2:
            #     csv_reader2 = csv.reader(csv_file2, delimiter=',')
            #     for i, line2 in enumerate(csv_reader2):
            #         if i == 0:
            #             continue
            #         psnr_grad_bounds.append(float(line2[0]))
            # array = np.array([psnr_bounds[0]])  # , psnr_bounds[2]])
            # grad_psnrs = np.array([psnr_bounds[1]])  # , psnr_bounds[3]])
            # name = './results/compare_siren/' + diff + fname.split('.')[0] + '.csv'
            # name2 = './results/compare_siren/' + diff + fname.split('.')[0] + '_grad.csv'
            # print('\nPred:', array)
            # print('\nGrad:', grad_psnrs)
            # np.savetxt(name2, grad_psnrs, delimiter=",")
            # np.savetxt(name, array, delimiter=",")
            # np.savetxt(name2, grad_psnrs, delimiter=",")


            if learn_bounds:
                import matplotlib.pyplot as plt

                b = m2.stages[0].middle_layers[0].bound.data.cpu().detach().numpy()
                f = torch.abs(m2.frequencies).max(1)[0].numpy()
                color = torch.abs(m2.frequencies).min(1)[0].numpy()
                # plt.scatter(f, b * boost, c=abs(color))
                plt.subplots(figsize=(3, 4))
                plt.scatter(f, b * boost, edgecolors='white')
                plt.ylim(0, 0.82)
                # plt.plot(28 * np.ones(10), np.linspace(0, 1, 10), 'g--')
                plt.xlabel('frequencies')
                plt.ylabel('bound')
                # plt.colorbar()
                plt.show()
