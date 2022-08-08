"""
波形のヒストグラムを作成する
"""

import argparse
import multiprocessing
import warnings
from functools import partial
from glob import glob
from pathlib import Path
from typing import Optional

import numpy
import yaml
from acoustic_feature_extractor.data.wave import Wave
from pytorch_trainer.iterators import MultiprocessIterator
from tqdm import tqdm
from yukarin_wavernn.config import Config
from yukarin_wavernn.data import encode_mulaw, encode_single


def _process(path: Path, bit: int, gaussian_noise_sigma: float):
    try:
        wave = Wave.load(path).wave
        assert wave.min() >= -1.0 and wave.max() <= 1.0

        if gaussian_noise_sigma > 0:
            wave += numpy.random.randn(*wave.shape) * gaussian_noise_sigma

        encoded = encode_single(encode_mulaw(wave, mu=2**bit), bit=bit)
        return numpy.histogram(encoded, bins=2**bit, range=(0, 2**bit))[0].astype(
            numpy.uint64
        )
    except:
        print(f"error!: {path} failed")
        raise


def count(
    input_wave_glob: str,
    bit: int,
    gaussian_noise_sigma: float,
    num_processes: Optional[int],
    output_path: Path,
):
    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    process = partial(_process, bit=bit, gaussian_noise_sigma=gaussian_noise_sigma)
    paths = [Path(p) for p in glob(input_wave_glob)]

    all_histogram = numpy.zeros(2**bit, dtype=numpy.uint64)

    with multiprocessing.Pool(processes=num_processes) as pool:
        it = pool.imap_unordered(process, paths, chunksize=2**6)
        for histogram in tqdm(it, desc="count", total=len(paths)):
            all_histogram += histogram

    print("min", "count", all_histogram.min(), "index", all_histogram.argmin())
    print("max", "count", all_histogram.max(), "index", all_histogram.argmax())

    numpy.save(output_path, all_histogram)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml_path", type=Path)
    parser.add_argument("--input_wave_glob", type=str)
    parser.add_argument("--bit", type=int)
    parser.add_argument("--gaussian_noise_sigma", type=float)
    parser.add_argument("--num_processes", type=int)
    parser.add_argument("--output_path", type=Path, default=Path("count.npy"))
    args = parser.parse_args()

    config: Optional[Config] = None
    if args.config_yaml_path:
        with args.config_yaml_path.open() as f:
            config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)

    input_wave_glob: str = (
        args.input_wave_glob
        if args.input_wave_glob is not None
        else config.dataset.input_wave_glob
    )
    bit: int = args.bit if args.bit is not None else config.dataset.bit_size
    gaussian_noise_sigma: float = (
        args.gaussian_noise_sigma
        if args.gaussian_noise_sigma is not None
        else config.dataset.gaussian_noise_sigma
    )

    num_processes: Optional[int] = args.num_processes
    output_path: Path = args.output_path

    count(
        input_wave_glob=input_wave_glob,
        bit=bit,
        gaussian_noise_sigma=gaussian_noise_sigma,
        num_processes=num_processes,
        output_path=output_path,
    )
