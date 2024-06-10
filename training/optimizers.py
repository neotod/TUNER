import torch
from mrnet.training.optimizer import OptimizationHandler


class ClampOptimizationHandler(OptimizationHandler):
    def __init__(self, model, optimizer, loss_function,
                 loss_weights, bound):
        super().__init__(model, optimizer, loss_function, loss_weights)
        device = self.get_device()
        self.bound = (bound / self.get_omega_0()).to(device)
        self.epoch = 0

    def _post_process(self, loss_dict):
        running_loss = super()._post_process(loss_dict)
        self.clamp()
        return running_loss

    def clamp(self):
        for name, weight in self.model.state_dict().items():
            if 'middle' in name and 'weight' in name:
                new_weight = torch.clamp(weight,
                                         min=-self.bound,
                                         max=self.bound)
                with torch.no_grad():
                    weight.copy_(new_weight)
        self.epoch += 1

    def get_device(self):
        dev = self.model.stages[-1].middle_layers[0].linear.weight.get_device()
        return torch.device(f'cuda:{dev}' if dev >= 0 else 'cpu')

    def get_omega_0(self):
        return self.model.stages[-1].middle_layers[0].omega_0
