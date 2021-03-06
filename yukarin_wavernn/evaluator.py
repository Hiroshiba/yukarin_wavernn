from pathlib import Path
from typing import Optional

import librosa
import numpy
from acoustic_feature_extractor.data.spectrogram import to_melcepstrum
from acoustic_feature_extractor.data.wave import Wave
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_wavernn.generator import Generator, SamplingPolicy

_logdb_const = 10.0 / numpy.log(10.0) * numpy.sqrt(2.0)


def _mcd(x: numpy.ndarray, y: numpy.ndarray) -> float:
    z = x - y
    r = numpy.sqrt((z * z).sum(axis=1)).mean()
    return _logdb_const * r


def calc_mcd(
    path1: Optional[Path] = None,
    path2: Optional[Path] = None,
    wave1: Optional[Wave] = None,
    wave2: Optional[Wave] = None,
):
    wave1 = Wave.load(path1) if wave1 is None else wave1
    wave2 = Wave.load(path2) if wave2 is None else wave2
    assert wave1.sampling_rate == wave2.sampling_rate

    sampling_rate = wave1.sampling_rate

    min_length = min(len(wave1.wave), len(wave2.wave))
    wave1.wave = wave1.wave[:min_length]
    wave2.wave = wave2.wave[:min_length]

    mc1 = to_melcepstrum(
        x=wave1.wave,
        sampling_rate=sampling_rate,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        order=24,
    )
    mc2 = to_melcepstrum(
        x=wave2.wave,
        sampling_rate=sampling_rate,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        order=24,
    )
    return _mcd(mc1, mc2)


def calc_silence_rate(
    path1: Optional[Path] = None,
    path2: Optional[Path] = None,
    wave1: Optional[Wave] = None,
    wave2: Optional[Wave] = None,
):
    wave1 = Wave.load(path1) if wave1 is None else wave1
    wave2 = Wave.load(path2) if wave2 is None else wave2
    assert wave1.sampling_rate == wave2.sampling_rate

    silence1 = ~librosa.effects._signal_to_frame_nonsilent(wave1.wave)
    silence2 = ~librosa.effects._signal_to_frame_nonsilent(wave2.wave)

    tp = numpy.logical_and(silence1, silence2).sum(dtype=float)
    tn = numpy.logical_and(~silence1, ~silence2).sum(dtype=float)
    fn = numpy.logical_and(silence1, ~silence2).sum(dtype=float)
    fp = numpy.logical_and(~silence1, silence2).sum(dtype=float)

    accuracy = (tp + tn) / (tp + tn + fn + fp)
    return accuracy


class GenerateEvaluator(nn.Module):
    def __init__(
        self,
        generator: Generator,
        time_length: float,
        local_padding_time_length: float,
        sampling_policy: SamplingPolicy = SamplingPolicy.random,
    ):
        super().__init__()
        self.generator = generator
        self.time_length = time_length
        self.local_padding_time_length = local_padding_time_length
        self.sampling_policy = sampling_policy

    def __call__(
        self,
        wave: Tensor,
        local: Optional[Tensor],
        speaker_num: Optional[Tensor] = None,
    ):
        batchsize = len(wave)

        wave_output = self.generator.generate(
            time_length=self.time_length + self.local_padding_time_length * 2,
            sampling_policy=self.sampling_policy,
            num_generate=batchsize,
            local_array=local,
            speaker_nums=speaker_num,
        )

        mcd_list = []
        sil_acc_list = []
        for wi, wo in zip(wave.cpu().numpy(), wave_output):
            wi = Wave(wave=wi, sampling_rate=wo.sampling_rate)

            if self.local_padding_time_length > 0:
                pad = int(wo.sampling_rate * self.local_padding_time_length)
                wo.wave = wo.wave[pad:-pad]

            mcd = calc_mcd(wave1=wi, wave2=wo)
            mcd_list.append(mcd)

            accuracy = calc_silence_rate(wave1=wi, wave2=wo)
            sil_acc_list.append(accuracy)

        scores = {
            "mcd": (numpy.mean(mcd_list), batchsize),
            "sil_acc": (numpy.mean(sil_acc_list), batchsize),
        }

        report(scores, self)
        return scores
