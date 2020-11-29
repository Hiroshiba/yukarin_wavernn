import argparse
import glob
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy
import yaml
from acoustic_feature_extractor.data.sampling_data import SamplingData
from more_itertools import chunked
from utility.save_arguments import save_arguments
from yukarin_wavernn.config import Config
from yukarin_wavernn.dataset import (
    SpeakerWavesDataset,
    TensorWrapperDataset,
    WavesDataset,
)
from yukarin_wavernn.dataset import create as create_dataset
from yukarin_wavernn.generator import Generator, SamplingPolicy


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def process(
    generator: Generator,
    local_paths: Sequence[Path],
    time_length: float,
    speaker_nums: Optional[Sequence[int]],
    sampling_policy: SamplingPolicy,
    output_dir: Path,
    postfix="",
):
    local_datas = [SamplingData.load(local_path) for local_path in local_paths]
    size = int((time_length + 5) * local_datas[0].rate)
    local_arrays = [
        local_data.array[:size]
        if len(local_data.array) >= size
        else numpy.pad(
            local_data.array,
            ((0, size - len(local_data.array)), (0, 0)),
            mode="edge",
        )
        for local_data in local_datas
    ]

    waves = generator.generate(
        time_length=time_length,
        sampling_policy=sampling_policy,
        num_generate=len(local_arrays),
        local_array=numpy.stack(local_arrays),
        speaker_nums=speaker_nums,
    )
    for wave, local_path in zip(waves, local_paths):
        wave.save(output_dir / (local_path.stem + postfix + ".wav"))


def generate(
    model_dir: Path,
    model_iteration: int,
    model_config: Path,
    time_length: float,
    input_batchsize: Optional[int],
    num_test: int,
    sampling_policy: SamplingPolicy,
    val_local_glob: str,
    val_speaker_num: Optional[int],
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    if model_config is None:
        model_config = model_dir / "config.yaml"
    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    print("model path: ", model_path)

    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=True,
    )

    batchsize = (
        input_batchsize if input_batchsize is not None else config.train.batchsize
    )

    dataset = create_dataset(config.dataset)["test"]
    if isinstance(dataset, TensorWrapperDataset):
        dataset = dataset.dataset

    if isinstance(dataset, WavesDataset):
        inputs = dataset.inputs
        local_paths = [input.path_local for input in inputs[:num_test]]
        speaker_nums = [None] * num_test
    elif isinstance(dataset, SpeakerWavesDataset):
        inputs = dataset.wave_dataset.inputs
        local_paths = [input.path_local for input in inputs[:num_test]]
        speaker_nums = dataset.speaker_nums[:num_test]
    else:
        raise ValueError(dataset)

    # random
    for local_path, speaker_num in zip(
        chunked(local_paths, batchsize), chunked(speaker_nums, batchsize)
    ):
        process(
            generator=generator,
            local_paths=local_path,
            time_length=time_length,
            speaker_nums=speaker_num if speaker_num[0] is not None else None,
            sampling_policy=sampling_policy,
            output_dir=output_dir,
        )

    # validation
    if val_local_glob is not None:
        local_paths = sorted([Path(p) for p in glob.glob(val_local_glob)])
        speaker_nums = [val_speaker_num] * len(local_paths)
        for local_path, speaker_num in zip(
            chunked(local_paths, batchsize), chunked(speaker_nums, batchsize)
        ):
            process(
                generator=generator,
                local_paths=local_path,
                time_length=time_length,
                speaker_nums=speaker_num if speaker_num[0] is not None else None,
                sampling_policy=sampling_policy,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--time_length", type=float, default=1)
    parser.add_argument("--input_batchsize", type=int)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument(
        "--sampling_policy", type=SamplingPolicy, default=SamplingPolicy.random
    )
    parser.add_argument("--val_local_glob")
    parser.add_argument("--val_speaker_num", type=int)
    parser.add_argument("--output_dir", type=Path)
    generate(**vars(parser.parse_args()))
