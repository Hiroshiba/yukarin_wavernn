from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Union

import numpy
import torch
import torch.nn.functional as F
from acoustic_feature_extractor.data.wave import Wave
from torch import Tensor
from tqdm import tqdm

from yukarin_wavernn.config import Config
from yukarin_wavernn.data import decode_mulaw, decode_single, encode_single
from yukarin_wavernn.model import create_predictor
from yukarin_wavernn.network.fast_forward import (
    fast_generate,
    get_fast_forward_params,
    to_numpy,
)
from yukarin_wavernn.network.wave_rnn import WaveRNN


class SamplingPolicy(str, Enum):
    random = "random"
    corrected_random = "corrected_random"
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
        max_batch_size: int = 10,
        use_fast_inference: bool = True,
    ):
        self.config = config
        self.max_batch_size = max_batch_size
        self.use_gpu = use_gpu
        self.use_fast_inference = use_fast_inference

        self.sampling_rate = config.dataset.sampling_rate
        self.mulaw = config.dataset.mulaw
        self.bit_size = config.dataset.bit_size
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

        if use_fast_inference and use_gpu:
            # setup cpp inference
            import yukarin_autoreg_cpp

            params = get_fast_forward_params(self.predictor)
            local_size = (
                config.network.conditioning_size * 2
                if config.network.conditioning_size is not None
                else 0
            )
            yukarin_autoreg_cpp.initialize(
                graph_length=1000,
                max_batch_size=max_batch_size,
                local_size=local_size,
                hidden_size=config.network.hidden_size,
                embedding_size=config.network.embedding_size,
                linear_hidden_size=config.network.linear_hidden_size,
                output_size=2 ** config.network.bit_size,
                **params,
            )

    def generate(
        self,
        time_length: float,
        sampling_policy: SamplingPolicy,
        num_generate: int,
        local_array: Union[numpy.ndarray, Tensor] = None,
        speaker_nums: Union[Sequence[int], Tensor] = None,
    ):
        assert num_generate <= self.max_batch_size
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

        if self.predictor.with_local:
            with torch.no_grad():
                local_array = self.predictor.forward_encode(
                    l_array=local_array, s_one=s_one
                )

        x = numpy.zeros(num_generate, dtype=numpy.float32)
        x = encode_single(x, bit=self.bit_size)

        hidden = numpy.zeros(
            (num_generate, self.predictor.gru.hidden_size),
            dtype=numpy.float32,
        )

        if sampling_policy == SamplingPolicy.corrected_random:
            low_probability_threshold = -18
        else:
            low_probability_threshold = -999

        if self.use_fast_inference and self.use_gpu:
            assert sampling_policy in [
                SamplingPolicy.random,
                SamplingPolicy.corrected_random,
            ]

            import yukarin_autoreg_cpp

            wave = numpy.zeros((length, num_generate), dtype=numpy.int32)
            yukarin_autoreg_cpp.inference(
                batch_size=num_generate,
                length=length,
                output=wave,
                x=x.astype(numpy.int32),
                l_array=to_numpy(local_array.transpose(0, 1)),
                hidden=to_numpy(hidden),
                low_probability_threshold=low_probability_threshold,
            )

        elif self.use_fast_inference and not self.use_gpu:
            assert sampling_policy == SamplingPolicy.random

            params = get_fast_forward_params(self.predictor)
            x_list = fast_generate(
                length=length,
                x=x,
                l_array=local_array.numpy(),
                h=hidden,
                low_probability_threshold=low_probability_threshold,
                **params,
            )
            wave = numpy.stack(x_list)
        else:
            with torch.no_grad():
                x = to_tensor(x).to(self.device)
                x_max = x
                hidden = to_tensor(hidden).to(self.device)
                x_list = []
                for i in tqdm(range(length), desc="generate"):
                    d_max, _ = self.predictor.forward_one(
                        prev_x=x_max, prev_l=local_array[:, i], hidden=hidden
                    )
                    d, hidden = self.predictor.forward_one(
                        prev_x=x, prev_l=local_array[:, i], hidden=hidden
                    )

                    if sampling_policy == SamplingPolicy.maximum:
                        is_random = False
                    else:
                        is_random = True
                        d[
                            F.log_softmax(d_max.double(), dim=1)
                            < low_probability_threshold
                        ] -= 200

                    x = self.predictor.sampling(d, maximum=not is_random)
                    x_max = self.predictor.sampling(d, maximum=True)
                    x_list.append(x)

                wave = torch.stack(x_list).cpu().numpy()

        wave = wave.T
        wave = decode_single(wave, bit=self.bit_size)
        if self.mulaw:
            wave = decode_mulaw(wave, mu=2 ** self.bit_size)

        return [Wave(wave=w_one, sampling_rate=self.sampling_rate) for w_one in wave]
