import os
from pathlib import Path
import torch

from networks.mrnet import Initializer
from training.tracking_logger import TrackTrainingListener
from training.optimizers import ClampOptimizationHandler
from mrnet.training.trainer import MRTrainer
from mrnet.training.optimizer import OptimizationHandler

from utils import (create_clamps, load_hyperparameters,
                   get_database)


if __name__ == "__main__":
    torch.manual_seed(777)
    psnrs_images = []
    for test_set in [True]:
        for fname in os.listdir("./data/kodak/"):
            psnr_bounds = psnr_grad_bounds = []
        
            # -- hyperparameters in configs --#
            hyper = load_hyperparameters("./configs/config_init.yml")
            project_name = hyper["project_name"] = "learn_bounds"
            hyper['data_path'] = "./data/kodak/" + fname
            train_dataset, test_dataset = get_database(hyper, False, 0.1)
            hf = hyper['hidden_features']
            # S Net --------------------------------------------------------- #
            hyper['omega_0'] = 256


            hyper["max_epochs_per_stage"] = 10
            torch.cuda.empty_cache()
            init_freqs = [[0, 1, 2, 3, 4, 5, 6, 7],
                          [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 255]]

            l1, l2 = len(init_freqs[0]), len(init_freqs[1])
            hf[0][0] = 4 * (l1 + l2 - 2) + 2 * ((l1 - 1) ** 2 + (l2 - 1) ** 2 + 1)
            hyper['block_limits'] = [int(max(init_freqs[1]) / 3)]
            learn_bounds = True
            hyper['hidden_features'] = hf
            initializer = Initializer(hyper, init_freqs=init_freqs,
                                      bias_init=True, init_W=test_set)
            m2 = initializer.get_model()
            bound = create_clamps(torch.tensor(hyper['bounds'][0]), hyper['block_limits'],
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
                    return torch.sin(x)
                m2.stages[0].middle_layers[0].forward = forward

                class BoundOptimizationHandler(OptimizationHandler):
                    def __init__(self, model, optimizer, loss_fn, weight_decay):
                        super().__init__(model, optimizer, loss_fn, weight_decay)
                        self.lam = (1 / (len(bound) * bound)).clone().cuda().requires_grad_(False)

                    def _post_process(self, loss_dict):
                        # with torch.no_grad():
                        b = self.model.stages[0].middle_layers[0].bound.data = \
                                torch.abs(self.model.stages[0].middle_layers[0].bound)

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

            if learn_bounds:
                import matplotlib.pyplot as plt

                b = m2.stages[0].middle_layers[0].bound.data.cpu().detach().numpy()
                f = torch.abs(m2.frequencies).max(1)[0].numpy()
                color = torch.abs(m2.frequencies).min(1)[0].numpy()
                plt.subplots(figsize=(3, 4))
                plt.scatter(f, b * boost, edgecolors='white')
                plt.ylim(0, 0.82)
                plt.xlabel('frequencies')
                plt.ylabel('bound')
                plt.show()
