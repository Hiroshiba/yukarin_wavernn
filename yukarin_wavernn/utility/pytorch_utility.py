from typing import Callable

import torch
from pytorch_trainer.dataset import convert
from pytorch_trainer.training.updaters.standard_updater import StandardUpdater
from torch import nn

try:
    from torch.cuda import amp
except ImportError:
    pass


def init_weights(model: torch.nn.Module, name: str):
    def _init_weights(layer: nn.Module):
        initializer: Callable
        if name == "uniform":
            initializer = torch.nn.init.uniform_
        elif name == "normal":
            initializer = torch.nn.init.normal_
        elif name == "xavier_uniform":
            initializer = torch.nn.init.xavier_uniform_
        elif name == "xavier_normal":
            initializer = torch.nn.init.xavier_normal_
        elif name == "kaiming_uniform":
            initializer = torch.nn.init.kaiming_uniform_
        elif name == "kaiming_normal":
            initializer = torch.nn.init.kaiming_normal_
        elif name == "orthogonal":
            initializer = torch.nn.init.orthogonal_
        elif name == "sparse":
            initializer = torch.nn.init.sparse_
        else:
            raise ValueError(name)

        for key, param in layer.named_parameters():
            if "weight" in key:
                initializer(param)

    model.apply(_init_weights)


class AmpUpdater(StandardUpdater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = amp.GradScaler()

    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        optimizer = self._optimizers["main"]
        model = self._models["main"]
        loss_func = self.loss_func or model

        for model in self._models.values():
            model.train()
        optimizer.zero_grad()

        with amp.autocast():
            if isinstance(in_arrays, tuple):
                loss = loss_func(*in_arrays)
            elif isinstance(in_arrays, dict):
                loss = loss_func(**in_arrays)
            else:
                loss = loss_func(in_arrays)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scaler"] = self.scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.scaler.load_state_dict(state_dict["scaler"])
