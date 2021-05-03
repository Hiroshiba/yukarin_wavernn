import unittest
from itertools import chain

import torch
from retry import retry
from torch.utils.data import ConcatDataset
from yukarin_wavernn.config import LocalNetworkType, LossConfig
from yukarin_wavernn.dataset import SpeakerWavesDataset
from yukarin_wavernn.model import Model
from yukarin_wavernn.network.wave_rnn import WaveRNN

from tests.utility import (
    DownLocalRandomDataset,
    LocalRandomDataset,
    RandomDataset,
    SignWaveDataset,
    setup_support,
    train_support,
)

sampling_rate = 8000
sampling_length = 880

gpu = 0
batch_size = 16
hidden_size = 896
iteration = 300

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def _create_model(
    local_size: int,
    local_scale: int = None,
    bit_size=10,
    speaker_size=0,
):
    network = WaveRNN(
        bit_size=bit_size,
        conditioning_size=128,
        embedding_size=256,
        use_wave_mask=False,
        hidden_size=hidden_size,
        linear_hidden_size=512,
        local_size=local_size,
        local_scale=local_scale if local_scale is not None else 1,
        local_layer_num=2,
        local_network_type=LocalNetworkType.gru,
        speaker_size=speaker_size,
        speaker_embedding_size=speaker_size // 4,
    )

    loss_config = LossConfig(
        silence_weight=0,
        mean_silence=True,
    )
    model = Model(loss_config=loss_config, predictor=network, local_padding_size=0)
    return model


def _get_trained_nll():
    return 5


class TestTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, bit=10, mulaw=True):
        model = _create_model(local_size=0)
        dataset = SignWaveDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

        def _first_hook(o):
            self.assertTrue(o["main/nll_coarse"] > trained_nll)

        def _last_hook(o):
            self.assertTrue(o["main/nll_coarse"] < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

        # save model
        torch.save(
            model.predictor.state_dict(),
            "/tmp/"
            f"test_training_wavernn"
            f"-bit={bit}"
            f"-mulaw={mulaw}"
            f"-speaker_size=0"
            f"-iteration={iteration}.pth",
        )

    def test_train(self):
        self._wrapper()


class TestCannotTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, bit=10, mulaw=False):
        model = _create_model(local_size=0)
        dataset = RandomDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

        def _first_hook(o):
            self.assertTrue(o["main/nll_coarse"] > trained_nll)

        def _last_hook(o):
            self.assertTrue(o["main/nll_coarse"] > trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        self._wrapper()


class TestLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, bit=10, mulaw=True):
        model = _create_model(local_size=2)
        dataset = LocalRandomDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

        def _first_hook(o):
            self.assertTrue(o["main/nll_coarse"] > trained_nll)

        def _last_hook(o):
            self.assertTrue(o["main/nll_coarse"] < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        self._wrapper()


class TestDownSampledLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, bit=10, mulaw=True):
        scale = 4

        model = _create_model(
            local_size=2 * scale,
            local_scale=scale,
        )
        dataset = DownLocalRandomDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            scale=scale,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

        def _first_hook(o):
            self.assertTrue(o["main/nll_coarse"] > trained_nll)

        def _last_hook(o):
            self.assertTrue(o["main/nll_coarse"] < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        self._wrapper()


class TestSpeakerTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, bit=10, mulaw=True):
        speaker_size = 4
        model = _create_model(
            local_size=0,
            speaker_size=speaker_size,
        )

        datasets = [
            SignWaveDataset(
                sampling_rate=sampling_rate,
                sampling_length=sampling_length,
                bit=bit,
                mulaw=mulaw,
                frequency=(i + 1) * 110,
            )
            for i in range(speaker_size)
        ]
        dataset = SpeakerWavesDataset(
            wave_dataset=ConcatDataset(datasets),
            speaker_nums=list(
                chain.from_iterable([i] * len(d) for i, d in enumerate(datasets))
            ),
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

        def _first_hook(o):
            self.assertTrue(o["main/nll_coarse"] > trained_nll)

        def _last_hook(o):
            self.assertTrue(o["main/nll_coarse"] < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

        # save model
        torch.save(
            model.predictor.state_dict(),
            "/tmp/"
            f"test_training_wavernn"
            f"-bit={bit}"
            f"-mulaw={mulaw}"
            f"-speaker_size={speaker_size}"
            f"-iteration={iteration}.pth",
        )

    def test_train(self):
        self._wrapper()
