import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from mrnet.training.listener import TrainingListener
from mrnet.training.optimizer import OptimizationHandler

from training.trainer import MRTrainer
from networks.mrnet import MRFactory
from utils import load_hyperparameters, get_database


class LearnBoundsOptimizationHandler(OptimizationHandler):

    def __init__(self, model, optimizer, loss_function, loss_weights):
        super().__init__(model, optimizer, loss_function, loss_weights)
        self.loss_function = self.make_d0_loss()

    def clamp(self):
        for name, weight in self.model.state_dict().items():
            if 'middle' in name and 'weight' in name:
                if not 'linear' in name: continue
                weight = torch.clamp(weight,
                                     min=-self.model.stages[0].first_layer.bounds,
                                     max=self.model.stages[0].first_layer.bounds)

    def _post_process(self, loss_dict):

        self.clamp()

        # b = self.model.stages[0].middle_layers[0].bounds
        self.model.stages[0].first_layer.bounds.data = torch.abs(self.model.stages[0].first_layer.bounds)
        print("Bounds - max:", self.model.stages[0].first_layer.bounds.max(), "\nmin:", self.model.stages[0].first_layer.bounds.min())
        print("Layer", self.model.stages[0].middle_layers[0].linear.weight.mean())
        running_loss = super()._post_process(loss_dict)
        return running_loss

    def make_d0_loss(self):
        def loss(output_dict, gt_dict, **kwargs):
            pred: torch.Tensor = output_dict['model_out']
            device = f"cuda:{pred.get_device()}" if pred.get_device() >= 0 else "cpu"
            device = kwargs.get('device', device)
            pred = pred.to(device)
            gt = gt_dict['d0'].to(device)

            loss_dict = {'d0': torch.nn.functional.mse_loss(pred, gt)}
            loss_dict['bound'] = torch.norm(self.model.stages[0].first_layer.bounds, p=1) / len(self.model.stages[0].first_layer.bounds)
        
            return loss_dict
        return loss

class BoundOptimizationHandler(OptimizationHandler):
    def __init__(self, model, optimizer, loss_fn, weight_decay):
        super().__init__(model, optimizer, loss_fn, weight_decay)

    def _post_process(self, loss_dict):
        # with torch.no_grad():
        b = self.model.stages[0].first_layer.bounds.data = \
                torch.abs(self.model.stages[0].first_layer.bounds)

        # loss_dict['bound'] = b @ self.lam
        loss_dict['bound'] = torch.norm(b, p=1) / len(b)
        print("Bounds\n max:", self.model.stages[0].first_layer.bounds.max(), "min:", self.model.stages[0].first_layer.bounds.min())
        running_loss = super()._post_process(loss_dict)
        return running_loss


if __name__ == "__main__":
    torch.manual_seed(777)

    # -- hyperparameters in configs --#
    hyper = load_hyperparameters(
        "/home/diana/taming/taming/configs/config_init.yml")
    hyper["learn_bounds"] = True
    hyper["bounds"] = [.5, .5]
    hyper["logger"] = "local"
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
    # print(m2.stages[0].middle_layers[0].linear.weight.max())
    # exit()

    with torch.no_grad():
        boost = 30
        m2.stages[0].middle_layers[0].bounds = \
            torch.nn.Parameter(m2.stages[0].middle_layers[0].bounds / boost)

    def forward(input):
        W = (torch.tanh(hyper['hidden_omega_0'] *
                        m2.stages[0].middle_layers[0].linear.weight) *
            boost * m2.stages[0].middle_layers[0].bounds)
        x = (input @ W.T + hyper['hidden_omega_0'] *
            m2.stages[0].middle_layers[0].linear.bias)  # UNDO
        # print(m2.stages[0].middle_layers[0].bound * boost)
        return torch.sin(x)
    m2.stages[0].middle_layers[0].forward = forward

    mrtrainer.train(hyper["device"])

    import matplotlib.pyplot as plt

    b = m2.stages[0].first_layer.bounds.data.cpu().detach().numpy()
    f = torch.abs(m2.stages[0].first_layer.linear.weight.detach().cpu() * hyper["period"] / (2 * np.pi)).max(1)[0].numpy()
    color = torch.abs(m2.stages[0].first_layer.linear.weight.detach().cpu() * hyper["period"] / (2 * np.pi)).min(1)[0].numpy()
    # plt.scatter(f, b * boost, c=abs(color))
    plt.subplots(figsize=(3, 4))
    plt.scatter(f, b * boost, edgecolors='white')
    # plt.ylim(0, 0.82)
    # plt.plot(28 * np.ones(10), np.linspace(0, 1, 10), 'g--')
    plt.xlabel('frequencies')
    plt.ylabel('bound')
    # plt.colorbar()
    plt.show()


# y = m2.stages[0].first_layer.bounds.detach().cpu().numpy() * hyper["omega_0"]
# x = torch.max(torch.abs(m2.stages[0].first_layer.linear.weight.detach().cpu()), dim=1)[0].numpy() * hyper["period"] / (2 * np.pi)
# c = torch.min(torch.abs(m2.stages[0].first_layer.linear.weight.detach().cpu()), dim=1)[0].numpy() * hyper["period"] / (2 * np.pi)
# plt.scatter(x, y, c=c)
# plt.show()