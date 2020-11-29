from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Union

import numpy
import torch
from acoustic_feature_extractor.data.wave import Wave
from torch import Tensor
from tqdm import tqdm

from yukarin_wavernn.config import Config
from yukarin_wavernn.data import decode_mulaw, decode_single, encode_single
from yukarin_wavernn.model import create_predictor
from yukarin_wavernn.network.wave_rnn import WaveRNN


class SamplingPolicy(str, Enum):
    random = "random"
    maximum = "maximum"
    # mix = "mix"


class MorphingPolicy(str, Enum):
    linear = "linear"
    sphere = "sphere"


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)

    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[WaveRNN, Path],
        use_gpu: bool,
    ):
        self.config = config

        self.sampling_rate = config.dataset.sampling_rate
        self.mulaw = config.dataset.mulaw
        self.bit_size = config.dataset.bit_size
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        time_length: float,
        sampling_policy: SamplingPolicy,
        num_generate: int,
        local_array: Union[numpy.ndarray, Tensor] = None,
        speaker_nums: Union[Sequence[int], Tensor] = None,
    ):
        assert local_array is None or len(local_array) == num_generate
        assert speaker_nums is None or len(speaker_nums) == num_generate

        length = int(self.sampling_rate * time_length)

        if local_array is None:
            local_array = torch.empty((num_generate, length, 0)).float()
        local_array = to_tensor(local_array).to(self.device)

        if speaker_nums is not None:
            speaker_nums = to_tensor(speaker_nums).reshape((-1,)).to(self.device)
            with torch.no_grad():
                s_one = self.predictor.forward_speaker(speaker_nums)
        else:
            s_one = None

        return self.main_forward(
            length=length,
            sampling_policy=sampling_policy,
            num_generate=num_generate,
            local_array=local_array,
            s_one=s_one,
        )

    def morphing(
        self,
        time_length: float,
        sampling_policy: SamplingPolicy,
        morphing_policy: MorphingPolicy,
        local_array1: Union[numpy.ndarray, Tensor],
        local_array2: Union[numpy.ndarray, Tensor],
        speaker_nums1: Sequence[int],
        speaker_nums2: Sequence[int],
        start_rates: Sequence[float],
        stop_rates: Sequence[float],
    ):
        raise NotImplementedError

        num_generate = len(local_array1)
        assert len(local_array2) == num_generate
        assert len(speaker_nums1) == num_generate
        assert len(speaker_nums2) == num_generate
        assert len(start_rates) == num_generate
        assert len(stop_rates) == num_generate

        length = int(self.sampling_rate * time_length)
        local_length = local_array1.shape[1]

        local_array1 = to_tensor(local_array1)
        local_array2 = to_tensor(local_array2)

        speaker_nums1 = to_tensor(numpy.asarray(speaker_nums1)).reshape((-1,))
        speaker_nums2 = to_tensor(numpy.asarray(speaker_nums2)).reshape((-1,))
        with torch.no_grad():
            s_one1 = numpy.repeat(
                self.predictor.forward_speaker(speaker_nums1)[:, None],
                local_length,
                axis=1,
            )
            s_one2 = numpy.repeat(
                self.predictor.forward_speaker(speaker_nums2)[:, None],
                local_length,
                axis=1,
            )

        # morphing
        start_rates = numpy.asarray(start_rates)
        stop_rates = numpy.asarray(stop_rates)
        morph_rates = to_tensor(
            numpy.linspace(
                start_rates,
                stop_rates,
                num=local_length,
                axis=1,
                dtype=numpy.float32,
            )
        ).reshape((num_generate, local_length, 1))

        local_array = local_array1 * morph_rates + local_array2 * (1 - morph_rates)

        if morphing_policy == MorphingPolicy.linear:
            s_one = s_one1 * morph_rates + s_one2 * (1 - morph_rates)
        elif morphing_policy == MorphingPolicy.sphere:
            omega = numpy.arccos(
                numpy.sum(
                    (s_one1 * s_one2)
                    / (
                        numpy.linalg.norm(s_one1, axis=2, keepdims=True)
                        * numpy.linalg.norm(s_one2, axis=2, keepdims=True)
                    ),
                    axis=2,
                    keepdims=True,
                )
            )
            sin_omega = numpy.sin(omega)
            s_one = (
                numpy.sin(morph_rates * omega) / sin_omega * s_one1
                + numpy.sin((1.0 - morph_rates) * omega) / sin_omega * s_one2
            ).astype(numpy.float32)
        else:
            raise ValueError(morphing_policy)

        return self.main_forward(
            length=length,
            sampling_policy=sampling_policy,
            num_generate=num_generate,
            local_array=local_array,
            s_one=s_one,
        )

    def main_forward(
        self,
        length: int,
        sampling_policy: SamplingPolicy,
        num_generate: int,
        local_array: Tensor,
        s_one: Tensor = None,
    ):
        if self.predictor.with_local:
            with torch.no_grad():
                local_array = self.predictor.forward_encode(
                    l_array=local_array, s_one=s_one
                )

        c = numpy.zeros([num_generate], dtype=numpy.float32)
        c = to_tensor(encode_single(c, bit=self.bit_size)).to(self.device)

        w_list = []
        hc = None
        for i in tqdm(range(length), desc="generate"):
            with torch.no_grad():
                c, hc = self.predictor.forward_one(
                    prev_x=c,
                    prev_l=local_array[:, i],
                    hidden=hc,
                )

            if sampling_policy == SamplingPolicy.random:
                is_random = True
            elif sampling_policy == SamplingPolicy.maximum:
                is_random = False
            else:
                raise ValueError(sampling_policy)

            c = self.predictor.sampling(c, maximum=not is_random)
            w_list.append(c)

        wave = torch.stack(w_list).cpu().numpy()

        wave = wave.T
        wave = decode_single(wave, bit=self.bit_size)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.bit_size)

        return [Wave(wave=w_one, sampling_rate=self.sampling_rate) for w_one in wave]
