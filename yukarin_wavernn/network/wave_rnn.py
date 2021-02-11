from typing import Optional

import numpy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from yukarin_wavernn.config import LocalNetworkType
from yukarin_wavernn.network.local_encoder import (
    ResidualBottleneckDilatedCNN,
    ResidualBottleneckDilatedCNNBN,
    SkipDilatedCNN,
)


def _call_1layer(gru: nn.GRU, x: Tensor, h: Optional[Tensor]):
    if h is not None:
        h = h.unsqueeze(0)
    y, h = gru(x, h)
    h = h.squeeze(0)
    return y, h


class WaveRNN(nn.Module):
    def __init__(
        self,
        bit_size: int,
        conditioning_size: int,
        embedding_size: int,
        hidden_size: int,
        linear_hidden_size: int,
        local_size: int,
        local_scale: int,
        local_layer_num: int,
        local_network_type: LocalNetworkType,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.bit_size = bit_size
        self.local_size = local_size
        self.local_scale = local_scale
        self.speaker_size = speaker_size
        self.speaker_embedder = (
            nn.Embedding(speaker_size, speaker_embedding_size)
            if self.with_speaker
            else None
        )

        input_size = local_size + (speaker_embedding_size if self.with_speaker else 0)

        local_gru: Optional[nn.Module] = None
        local_encoder: Optional[nn.Module] = None
        if self.with_local:
            if local_network_type == LocalNetworkType.gru:
                local_gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=conditioning_size,
                    num_layers=local_layer_num,
                    batch_first=True,
                    bidirectional=True,
                )
            elif local_network_type == LocalNetworkType.skip_dilated_cnn:
                local_encoder = SkipDilatedCNN(
                    input_size=input_size,
                    layer_num=local_layer_num,
                    conditioning_size=conditioning_size,
                )
            elif local_network_type == LocalNetworkType.residual_bottleneck_dilated_cnn:
                local_encoder = ResidualBottleneckDilatedCNN(
                    input_size=input_size,
                    layer_num=local_layer_num,
                    conditioning_size=conditioning_size,
                )
            elif (
                local_network_type
                == LocalNetworkType.residual_bottleneck_dilated_cnn_bn
            ):
                local_encoder = ResidualBottleneckDilatedCNNBN(
                    input_size=input_size,
                    layer_num=local_layer_num,
                    conditioning_size=conditioning_size,
                )
            else:
                raise ValueError(local_network_type)
        self.local_gru = local_gru
        self.local_encoder = local_encoder

        self.x_embedder = nn.Embedding(self.bins, embedding_size)

        in_size = embedding_size + (
            (
                2 * conditioning_size
                if local_network_type == LocalNetworkType.gru
                else conditioning_size
            )
            if self.with_local
            else 0
        )
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.O1 = nn.Linear(hidden_size, linear_hidden_size)
        self.O2 = nn.Linear(linear_hidden_size, self.bins)

    @property
    def bins(self):
        return 2 ** self.bit_size

    @property
    def with_speaker(self):
        return self.speaker_size > 0

    @property
    def with_local(self):
        return self.local_size > 0 or self.with_speaker

    def forward(
        self,
        x_array: Tensor,
        l_array: Tensor,
        s_one: Optional[Tensor] = None,
        local_padding_size: int = 0,
        hidden: Optional[Tensor] = None,
    ):
        """
        x: wave
        l: local
        s: speaker
        :param x_array: int (batch_size, N+1)
        :param l_array: float (batch_size, lN, ?)
        :param s_one: int (batch_size, )
        :param local_padding_size:
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_x_array: float (batch_size, ?, N)
            hidden: float (batch_size, hidden_size)
        """
        assert (
            l_array.shape[2] == self.local_size
        ), f"{l_array.shape[2]} {self.local_size}"

        if self.with_speaker:
            s_one = self.forward_speaker(s_one)

        l_array = self.forward_encode(
            l_array=l_array, s_one=s_one
        )  # (batch_size, N + pad, ?)
        if local_padding_size > 0:
            l_array = l_array[
                :, local_padding_size:-local_padding_size
            ]  # (batch_size, N, ?)

        out_x_array, hidden = self.forward_rnn(
            x_array=x_array[:, :-1],
            l_array=l_array[:, 1:],
            hidden=hidden,
        )
        return out_x_array, hidden

    def forward_speaker(self, s_one: Tensor):
        """
        :param s_one: int (batch_size, )
        :return:
            s_one: float (batch_size, ?)
        """
        s_one = self.speaker_embedder(s_one)
        return s_one

    def forward_encode(
        self,
        l_array: Tensor,
        s_one: Optional[Tensor] = None,
    ):
        """
        :param l_array: float (batch_size, lN, ?)
        :param s_one: float (batch_size, ?) or (batch_size, lN, ?)
        :return:
            l_array: float (batch_size, N, ?)
        """
        if not self.with_local:
            return l_array

        length = l_array.shape[1]  # lN

        if self.with_speaker:
            if s_one.ndim == 2:
                s_one = s_one.unsqueeze(dim=1)  # shape: (batch_size, 1, ?)
                s_array = s_one.expand(
                    s_one.shape[0], length, s_one.shape[2]
                )  # shape: (batch_size, lN, ?)
            else:
                s_array = s_one  # shape: (batch_size, lN, ?)
            l_array = torch.cat((l_array, s_array), dim=2)  # (batch_size, lN, ?)

        if self.local_gru is not None:
            l_array, _ = self.local_gru(l_array)
        elif self.local_encoder is not None:
            l_array = self.local_encoder(l_array)
        else:
            raise ValueError

        l_array = l_array.repeat_interleave(
            self.local_scale, dim=1
        )  # shape: (batch_size, N, ?)
        return l_array

    def forward_rnn(
        self,
        x_array: Tensor,
        l_array: Tensor,
        hidden: Tensor = None,
    ):
        """
        :param x_array: int (batch_size, N)
        :param l_array: (batch_size, N, ?)
        :param hidden: float (batch_size, hidden_size)
        :return:
            out_x_array: float (batch_size, ?, N)
            hidden: float (batch_size, hidden_size)
        """
        assert (
            x_array.shape == l_array.shape[:2]
        ), f"{x_array.shape}, {l_array.shape[:2]}"

        batch_size = x_array.shape[0]
        length = x_array.shape[1]  # N

        x_array = x_array.reshape([batch_size * length, 1])  # (batchsize * N, 1)
        x_array = self.x_embedder(x_array).reshape(
            [batch_size, length, -1]
        )  # (batch_size, N, ?)

        xl_array = torch.cat((x_array, l_array), dim=2)  # (batch_size, N, ?)

        out_hidden, new_hidden = _call_1layer(
            self.gru, xl_array, hidden
        )  # (batch_size, N, hidden_size)

        out_hidden = out_hidden.reshape(batch_size * length, -1)  # (batch_size * N, ?)
        out_x_array = self.O2(F.relu(self.O1(out_hidden)))  # (batch_size * N, ?)
        out_x_array = out_x_array.reshape(batch_size, length, -1)  # (batch_size, N, ?)
        out_x_array = out_x_array.transpose(1, 2)  # (batch_size, ?, N)

        return out_x_array, new_hidden

    def forward_one(
        self,
        prev_x: Tensor,
        prev_l: Tensor,
        hidden: Tensor = None,
    ):
        """
        :param prev_x: int (batch_size, )
        :param prev_l: (batch_size, ?)
        :param hidden: float (batch_size, single_hidden_size)
        :return:
            out_x: float (batch_size, ?)
            hidden: float (batch_size, hidden_size)
        """
        batch_size = prev_x.shape[0]

        prev_x = self.x_embedder(prev_x[:, numpy.newaxis]).reshape(
            [batch_size, -1]
        )  # (batch_size, ?)

        prev_xl = torch.cat((prev_x, prev_l), dim=1).unsqueeze(1)  # (batch_size, 1, ?)

        out_hidden, new_hidden = _call_1layer(
            self.gru,
            prev_xl,
            hidden,
        )  # (batch_size, single_hidden_size)
        out_hidden = out_hidden.squeeze(1)

        out_x = self.O2(F.relu(self.O1(out_hidden)))  # (batch_size, ?)

        return out_x, new_hidden

    def sampling(self, dist: Tensor, maximum=True):
        if maximum:
            sampled = torch.argmax(dist, dim=1)
        else:
            prob = F.log_softmax(dist.double(), dim=1)
            rand = torch.from_numpy(numpy.random.gumbel(size=dist.shape)).to(
                dist.device
            )
            sampled = torch.argmax(prob + rand, dim=1)
        return sampled
