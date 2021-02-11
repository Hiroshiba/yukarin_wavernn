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


class ResidualBottleneckDilatedCNN(nn.Module):
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

        self.conv1_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=conditioning_size,
                        out_channels=conditioning_size // 4,
                        kernel_size=1,
                    )
                )
                for _ in range(layer_num)
            ]
        )
        self.conv2_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=conditioning_size // 4,
                        out_channels=conditioning_size // 4,
                        kernel_size=3,
                        dilation=2 ** i,
                        padding=2 ** i,
                    )
                )
                for i in range(layer_num)
            ]
        )
        self.conv3_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=conditioning_size // 4,
                        out_channels=conditioning_size,
                        kernel_size=1,
                    )
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, lN, ?)
        """
        h = x.transpose(1, 2)
        h = self.pre(h)
        for conv1, conv2, conv3 in zip(
            self.conv1_list, self.conv2_list, self.conv3_list
        ):
            h = h + conv3(F.relu(conv2(F.relu(conv1(F.relu(h))))))
        h = h.transpose(1, 2)
        return h


class ResidualBottleneckDilatedCNNBN(nn.Module):
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

        self.bn1_list = nn.ModuleList(
            [nn.BatchNorm1d(conditioning_size) for _ in range(layer_num)]
        )
        self.conv1_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=conditioning_size,
                    out_channels=conditioning_size // 4,
                    kernel_size=1,
                )
                for _ in range(layer_num)
            ]
        )
        self.bn2_list = nn.ModuleList(
            [nn.BatchNorm1d(conditioning_size // 4) for _ in range(layer_num)]
        )
        self.conv2_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=conditioning_size // 4,
                    out_channels=conditioning_size // 4,
                    kernel_size=3,
                    dilation=2 ** i,
                    padding=2 ** i,
                )
                for i in range(layer_num)
            ]
        )
        self.bn3_list = nn.ModuleList(
            [nn.BatchNorm1d(conditioning_size // 4) for _ in range(layer_num)]
        )
        self.conv3_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=conditioning_size // 4,
                    out_channels=conditioning_size,
                    kernel_size=1,
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, lN, ?)
        """
        h = x.transpose(1, 2)
        h = self.pre(h)
        for bn1, conv1, bn2, conv2, bn3, conv3 in zip(
            self.bn1_list,
            self.conv1_list,
            self.bn2_list,
            self.conv2_list,
            self.bn3_list,
            self.conv3_list,
        ):
            h = h + conv3(F.relu(bn3(conv2(F.relu(bn2(conv1(F.relu(bn1(h)))))))))
        h = h.transpose(1, 2)
        return h
