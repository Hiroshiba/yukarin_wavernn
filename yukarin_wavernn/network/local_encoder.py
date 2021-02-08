import torch.nn.functional as F
from torch import Tensor, nn


class SkipDilatedCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        layer_num: int,
        conditioning_size: int,
    ):
        super().__init__()

        self.pre = nn.Conv1d(
            in_channels=input_size, out_channels=conditioning_size, kernel_size=1
        )

        self.conv_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=conditioning_size,
                        out_channels=conditioning_size,
                        kernel_size=3,
                        dilation=2 ** i,
                        padding=2 ** i,
                    )
                )
                for i in range(layer_num)
            ]
        )

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, lN, ?)
        """
        h = x.transpose(1, 2)
        h = self.pre(h)
        for conv in self.conv_list:
            h = h + conv(F.relu(h))
        h = h.transpose(1, 2)
        return h
