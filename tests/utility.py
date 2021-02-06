from pathlib import Path
from typing import Callable, Dict

import numpy
from pytorch_trainer import Reporter
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training.updaters import StandardUpdater
from torch import device, optim
from torch.utils.data.dataset import Dataset
from yukarin_wavernn.config import LocalNetworkType
from yukarin_wavernn.dataset import BaseWaveDataset, TensorWrapperDataset
from yukarin_wavernn.model import Model


def get_data_directory() -> Path:
    return Path(__file__).parent.relative_to(Path.cwd()) / "data"


class RandomDataset(BaseWaveDataset):
    def __len__(self):
        return 100

    def __getitem__(self, i):
        length = self.sampling_length
        wave = numpy.random.rand(length) * 2 - 1
        local = numpy.empty(shape=(length, 0), dtype=numpy.float32)
        silence = numpy.zeros(shape=(length,), dtype=numpy.bool)
        return self.convert_to_dict(wave, silence, local)


class LocalRandomDataset(RandomDataset):
    def __getitem__(self, i):
        d = super().__getitem__(i)
        d["local"] = numpy.stack(
            (
                d["encoded_coarse"].astype(numpy.float32) / 256,
                d["encoded_coarse"].astype(numpy.float32) / 256,
            ),
            axis=1,
        )
        return d


class DownLocalRandomDataset(LocalRandomDataset):
    def __init__(self, scale: int, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def __getitem__(self, i):
        d = super().__getitem__(i)
        l = numpy.reshape(d["local"], (-1, self.scale * d["local"].shape[1]))
        l[numpy.isnan(l)] = 0
        d["local"] = l
        return d


class SignWaveDataset(BaseWaveDataset):
    def __init__(
        self,
        sampling_rate: int,
        sampling_length: int,
        bit: int,
        mulaw: bool,
        frequency: float = 440,
    ):
        super().__init__(
            sampling_length=sampling_length,
            bit=bit,
            mulaw=mulaw,
            local_sampling_rate=None,
            local_padding_size=0,
        )
        self.sampling_rate = sampling_rate
        self.frequency = frequency

    def __len__(self):
        return 100

    def __getitem__(self, i):
        rate = self.sampling_rate
        length = self.sampling_length
        rand = numpy.random.rand()

        wave = numpy.sin(
            (numpy.arange(length) * self.frequency / rate + rand) * 2 * numpy.pi
        )
        local = numpy.empty(shape=(length, 0), dtype=numpy.float32)
        silence = numpy.zeros(shape=(length,), dtype=numpy.bool)

        d = self.convert_to_dict(wave, silence, local)
        return d


def _create_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer


def setup_support(
    batch_size: int,
    device: device,
    model: Model,
    dataset: Dataset,
):
    model.to(device)

    optimizer = _create_optimizer(model)
    train_iter = MultiprocessIterator(TensorWrapperDataset(dataset), batch_size)

    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    reporter = Reporter()
    reporter.add_observer("main", model)

    return updater, reporter


def train_support(
    iteration: int,
    reporter: Reporter,
    updater: StandardUpdater,
    first_hook: Callable[[Dict], None] = None,
    last_hook: Callable[[Dict], None] = None,
):
    observation: Dict = {}
    for i in range(iteration):
        with reporter.scope(observation):
            updater.update()

        if i % 100 == 0:
            print(observation)

        if i == 0:
            if first_hook is not None:
                first_hook(observation)

    print(observation)
    if last_hook is not None:
        last_hook(observation)


def get_test_config(
    bit,
    mulaw,
    speaker_size,
):
    dataset_config = type(
        "DatasetConfig",
        (object,),
        dict(
            sampling_rate=8000,
            mulaw=mulaw,
            bit_size=bit,
        ),
    )
    network_config = type(
        "NetworkConfig",
        (object,),
        dict(
            bit_size=bit,
            hidden_size=896,
            local_size=0,
            conditioning_size=128,
            embedding_size=256,
            linear_hidden_size=512,
            local_scale=1,
            local_layer_num=2,
            local_network_type=LocalNetworkType.gru,
            speaker_size=speaker_size,
            speaker_embedding_size=speaker_size // 4,
        ),
    )
    config = type(
        "Config",
        (object,),
        dict(
            dataset=dataset_config,
            network=network_config,
        ),
    )
    return config


def get_test_model_path(
    bit,
    mulaw,
    speaker_size,
    iteration,
):
    return Path(
        f"tests/data/test_training_wavernn"
        f"-bit={bit}"
        f"-mulaw={mulaw}"
        f"-speaker_size={speaker_size}"
        f"-iteration={iteration}.pth"
    )
