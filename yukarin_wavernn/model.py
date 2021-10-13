from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_wavernn.config import LossConfig
from yukarin_wavernn.network.wave_rnn import WaveRNN


class Model(nn.Module):
    def __init__(
        self,
        loss_config: LossConfig,
        predictor: WaveRNN,
        local_padding_size: int,
    ):
        super().__init__()
        self.loss_config = loss_config
        self.predictor = predictor
        self.local_padding_size = local_padding_size

    def __call__(
        self,
        coarse: Tensor,
        encoded_coarse: Tensor,
        local: Tensor,
        silence: Tensor,
        randomed_encoded_coarse: Optional[Tensor] = None,
        speaker_num: Optional[Tensor] = None,
    ):
        x_array = (
            encoded_coarse
            if randomed_encoded_coarse is None
            else randomed_encoded_coarse
        )

        out_c_array, _ = self.predictor(
            x_array=x_array,
            l_array=local,
            s_one=speaker_num,
            local_padding_size=self.local_padding_size,
        )

        target_coarse = encoded_coarse[:, 1:]
        nll_coarse = F.cross_entropy(out_c_array, target_coarse, reduction="none")

        silence_weight = self.loss_config.silence_weight
        if silence_weight == 0:
            nll_coarse = nll_coarse[~silence]
        elif silence_weight < 0:
            nll_coarse = nll_coarse[~silence] + nll_coarse[silence] * silence_weight

        nll_coarse = (
            torch.mean(nll_coarse)
            if self.loss_config.mean_silence
            else torch.sum(nll_coarse) / silence.size
        )

        loss = nll_coarse
        losses = dict(loss=loss, nll_coarse=nll_coarse)

        if not self.training:
            losses = {key: (l, len(coarse)) for key, l in losses.items()}  # add weight
        report(losses, self)
        return loss
