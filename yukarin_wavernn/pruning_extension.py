"""
WaveRNN的な枝刈り
"""

from typing import Optional

import torch
from pytorch_trainer.training import Extension, Trainer
from torch import Tensor


class PruningExtension(Extension):
    def __init__(
        self,
        weight: Tensor,
        density: float,
        with_diag: bool,
        t_start: int,
        t_end: int,
        interval: int,
    ):
        self.weight = weight
        self.density = density
        self.with_diag = with_diag
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval

        self.t = 0
        self._mask: Optional[Tensor] = None

        assert (
            weight.ndim == 2
            and weight.shape[0] == weight.shape[1]
            and weight.shape[0] % 16 == 0
        )

    @torch.no_grad()
    def __call__(self, trainer: Trainer):
        self.t += 1

        t = self.t
        density = self.density
        with_diag = self.with_diag
        t_start = self.t_start
        t_end = self.t_end
        interval = self.interval
        weight = self.weight

        if t < t_start or (t - t_start) % interval != 0:
            return

        weight = weight.transpose(1, 0)
        A = weight
        N = A.shape[0]
        if t < t_end:
            r = 1.0 - (t - t_start) / (t_end - t_start)
            density = 1 - (1 - density) * (1 - r * r * r)

        if with_diag:
            A = A - torch.diag(torch.diag(A))

        L = torch.reshape(A, (N, N // 16, 16))
        S = torch.sum(L * L, axis=-1)
        SS, _ = torch.sort(torch.reshape(S, (-1,)))
        thresh = SS[round(N * N // 16 * (1 - density))]
        mask = (S >= thresh).float()
        mask = torch.repeat_interleave(mask, 16, axis=1)

        if with_diag:
            mask.add_(torch.eye(N).to(A.device)).clamp_(max=1)

        weight *= mask.detach()

        self._mask = mask.cpu().clone().to(bool)

    def state_dict(self):
        return {"t": self.t, "mask": self._mask}

    def load_state_dict(self, state_dict):
        self.t = state_dict["t"]
        self._mask = state_dict["mask"]
