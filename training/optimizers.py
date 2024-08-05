import torch
from mrnet.training.optimizer import OptimizationHandler


class ClampOptimizationHandler(OptimizationHandler):
    def __init__(self, model, optimizer, loss_function,
                 loss_weights, bound):
        super().__init__(model, optimizer, loss_function, loss_weights)
        self.bounds = self.prepare_bounds(bound)
        self.epoch = 0

    def _post_process(self, loss_dict):
        running_loss = super()._post_process(loss_dict)
        self.clamp()
        return running_loss

    def clamp(self):
        i = 0
        for name, weight in self.model.state_dict().items():
            if 'middle' in name and 'weight' in name:
                b = self.bounds[i]
                new_weight = torch.clamp(weight,
                                         min=-b,
                                         max=b)
                with torch.no_grad():
                    weight.copy_(new_weight)
                i += 1
        self.epoch += 1

    def get_device(self):
        dev = self.model.stages[-1].middle_layers[0].linear.weight.get_device()
        return torch.device(f'cuda:{dev}' if dev >= 0 else 'cpu')

    def get_omega_0(self):
        return self.model.stages[-1].middle_layers[0].omega_0

    def prepare_bounds(self, bound):
        dev = self.get_device()
        n_bounds = len(self.model.stages[-1].middle_layers)
        if isinstance(bound, float) and bound > 0:
            b = torch.tensor(bound) / self.get_omega_0()
            bounds = [b.to(dev) for _ in range(n_bounds)]
        elif len(bound) == n_bounds:
            bounds = [(torch.tensor(b) / self.get_omega_0()).to(dev)
                      for b in bound]
        else:
            raise ValueError("Bounds must be positive")
        return bounds
