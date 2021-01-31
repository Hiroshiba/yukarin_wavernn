import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from yukarin_wavernn.config import DatasetConfig
from yukarin_wavernn.data import encode_mulaw, encode_single


@dataclass
class Input:
    wave: Wave
    silence: SamplingData
    local: SamplingData


@dataclass
class LazyInput:
    path_wave: Path
    path_silence: Path
    path_local: Path

    def generate(self):
        return Input(
            wave=Wave.load(self.path_wave),
            silence=SamplingData.load(self.path_silence),
            local=SamplingData.load(self.path_local),
        )


class BaseWaveDataset(Dataset):
    def __init__(
        self,
        sampling_length: int,
        bit: int,
        mulaw: bool,
        local_sampling_rate: Optional[int],
        local_padding_size: int,
    ):
        self.sampling_length = sampling_length
        self.bit = bit
        self.mulaw = mulaw
        self.local_sampling_rate = local_sampling_rate
        self.local_padding_size = local_padding_size

    @staticmethod
    def extract_input(
        sampling_length: int,
        wave_data: Wave,
        silence_data: SamplingData,
        local_data: SamplingData,
        local_sampling_rate: Optional[int],
        local_padding_size: int,
        padding_value=0,
    ):
        """
        :return:
            wave: (sampling_length, )
            silence: (sampling_length, )
            local: (sampling_length // scale + pad, )
        """
        sr = wave_data.sampling_rate
        sl = sampling_length

        if local_sampling_rate is None:
            l_rate = local_data.rate
            l_array = local_data.array
        else:
            l_rate = local_sampling_rate
            l_array = local_data.resample(l_rate)

        assert sr % l_rate == 0
        l_scale = int(sr // l_rate)

        length = len(l_array) * l_scale
        assert (
            abs(length - len(wave_data.wave)) < l_scale * 4
        ), f"{abs(length - len(wave_data.wave))} {l_scale}"

        assert local_padding_size % l_scale == 0
        l_pad = local_padding_size // l_scale

        l_length = length // l_scale
        l_sl = sl // l_scale

        for _ in range(10000):
            if l_length > l_sl:
                l_offset = numpy.random.randint(l_length - l_sl)
            else:
                l_offset = 0
            offset = l_offset * l_scale

            silence = numpy.squeeze(silence_data.resample(sr, index=offset, length=sl))
            if not silence.all():
                break
        else:
            raise Exception("cannot pick not silence data")

        wave = wave_data.wave[offset : offset + sl]

        # local
        l_start, l_end = l_offset - l_pad, l_offset + l_sl + l_pad
        if l_start < 0 or l_end > l_length:
            shape = list(l_array.shape)
            shape[0] = l_sl + l_pad * 2
            local = numpy.ones(shape=shape, dtype=l_array.dtype) * padding_value
            if l_start < 0:
                p_start = -l_start
                l_start = 0
            else:
                p_start = 0
            if l_end > l_length:
                p_end = l_sl + l_pad * 2 - (l_end - l_length)
                l_end = l_length
            else:
                p_end = l_sl + l_pad * 2
            local[p_start:p_end] = l_array[l_start:l_end]
        else:
            local = l_array[l_start:l_end]

        return wave, silence, local

    @staticmethod
    def add_noise(wave: numpy.ndarray, gaussian_noise_sigma: float):
        if gaussian_noise_sigma > 0:
            wave += numpy.random.normal(0, gaussian_noise_sigma)
            wave[wave > 1.0] = 1.0
            wave[wave < -1.0] = -1.0
        return wave

    def convert_to_dict(
        self, wave: numpy.ndarray, silence: numpy.ndarray, local: numpy.ndarray
    ):
        if self.mulaw:
            wave = encode_mulaw(wave, mu=2 ** self.bit)
        encoded_coarse = encode_single(wave, bit=self.bit)
        coarse = wave.ravel().astype(numpy.float32)
        return dict(
            coarse=coarse,
            encoded_coarse=encoded_coarse,
            local=local,
            silence=silence[1:],
        )

    def make_input(
        self,
        wave_data: Wave,
        silence_data: SamplingData,
        local_data: SamplingData,
        gaussian_noise_sigma: float,
    ):
        wave, silence, local = self.extract_input(
            sampling_length=self.sampling_length,
            wave_data=wave_data,
            silence_data=silence_data,
            local_data=local_data,
            local_sampling_rate=self.local_sampling_rate,
            local_padding_size=self.local_padding_size,
        )
        wave = self.add_noise(wave=wave, gaussian_noise_sigma=gaussian_noise_sigma)
        d = self.convert_to_dict(wave, silence, local)
        return d


class WavesDataset(BaseWaveDataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        sampling_length: int,
        bit: int,
        mulaw: bool,
        local_sampling_rate: Optional[int],
        local_padding_size: int,
        gaussian_noise_sigma: float,
    ):
        super().__init__(
            sampling_length=sampling_length,
            bit=bit,
            mulaw=mulaw,
            local_sampling_rate=local_sampling_rate,
            local_padding_size=local_padding_size,
        )
        self.inputs = inputs
        self.gaussian_noise_sigma = gaussian_noise_sigma

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.make_input(
            wave_data=input.wave,
            silence_data=input.silence,
            local_data=input.local,
            gaussian_noise_sigma=self.gaussian_noise_sigma,
        )


class NonEncodeWavesDataset(Dataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        time_length: float,
        local_sampling_rate: Optional[int],
        local_padding_time_length: float,
    ):
        self.inputs = inputs
        self.time_length = time_length
        self.local_sampling_rate = local_sampling_rate
        self.local_padding_time_length = local_padding_time_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        sampling_length = int(input.wave.sampling_rate * self.time_length)
        local_padding_size = int(
            input.wave.sampling_rate * self.local_padding_time_length
        )

        wave, silence, local = BaseWaveDataset.extract_input(
            sampling_length=sampling_length,
            wave_data=input.wave,
            silence_data=input.silence,
            local_data=input.local,
            local_sampling_rate=self.local_sampling_rate,
            local_padding_size=local_padding_size,
        )
        return dict(
            wave=wave,
            local=local,
        )


class SpeakerWavesDataset(Dataset):
    def __init__(self, wave_dataset: Dataset, speaker_nums: List[int]):
        assert len(wave_dataset) == len(speaker_nums)
        self.wave_dataset = wave_dataset
        self.speaker_nums = speaker_nums

    def __len__(self):
        return len(self.wave_dataset)

    def __getitem__(self, i):
        d = self.wave_dataset[i]
        d["speaker_num"] = numpy.array(self.speaker_nums[i], dtype=numpy.int64)
        return d


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def create(config: DatasetConfig):
    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_wave_glob))}
    fn_list = sorted(wave_paths.keys())
    assert len(fn_list) > 0

    silence_paths = {
        Path(p).stem: Path(p) for p in glob.glob(str(config.input_silence_glob))
    }
    assert set(fn_list) == set(silence_paths.keys())

    local_paths = {
        Path(p).stem: Path(p) for p in glob.glob(str(config.input_local_glob))
    }
    assert set(fn_list) == set(local_paths.keys())

    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.load(
            open(config.speaker_dict_path)
        )
        assert config.num_speaker == len(fn_each_speaker)

        speaker_nums = {
            fn: speaker_num
            for speaker_num, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_nums.keys()))
    else:
        speaker_nums = None

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    num_test = config.num_test
    num_train = (
        config.num_train if config.num_train is not None else len(fn_list) - num_test
    )

    trains = fn_list[num_test:][:num_train]
    tests = fn_list[:num_test]

    def Dataset(fns, for_evaluate=False):
        inputs = [
            LazyInput(
                path_wave=wave_paths[fn],
                path_silence=silence_paths[fn],
                path_local=local_paths[fn],
            )
            for fn in fns
        ]

        if not for_evaluate:
            dataset = WavesDataset(
                inputs=inputs,
                sampling_length=config.sampling_length,
                bit=config.bit_size,
                mulaw=config.mulaw,
                local_sampling_rate=config.local_sampling_rate,
                local_padding_size=config.local_padding_size,
                gaussian_noise_sigma=config.gaussian_noise_sigma,
            )
        else:
            dataset = NonEncodeWavesDataset(
                inputs=inputs,
                time_length=config.time_length_evaluate,
                local_sampling_rate=config.local_sampling_rate,
                local_padding_time_length=config.local_padding_time_length_evaluate,
            )

        if speaker_nums is not None:
            dataset = SpeakerWavesDataset(
                wave_dataset=dataset,
                speaker_nums=[speaker_nums[fn] for fn in fns],
            )

        dataset = TensorWrapperDataset(dataset)

        if for_evaluate:
            dataset = ConcatDataset([dataset] * config.num_times_evaluate)

        return dataset

    return {
        "train": Dataset(trains),
        "test": Dataset(tests),
        "eval": Dataset(tests, for_evaluate=True),
    }
