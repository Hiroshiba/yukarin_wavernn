import argparse
import time
from pathlib import Path

import numpy
import yaml
from yukarin_wavernn.config import Config
from yukarin_wavernn.generator import Generator, SamplingPolicy


def speed_check(
    model_path: Path,
    model_config_path: Path,
    time_length: float,
    batchsize: int,
    sampling_policy: str,
    use_gpu: bool,
):
    config = Config.from_dict(yaml.safe_load(model_config_path.open()))

    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
        max_batch_size=batchsize,
        use_fast_inference=True,
    )
    local_length = config.dataset.sampling_rate // config.network.local_scale
    local_size = config.network.local_size
    local_array = numpy.random.rand(batchsize, local_length, local_size).astype(
        numpy.float32
    )

    start = time.time()

    generator.generate(
        time_length=time_length,
        sampling_policy=SamplingPolicy(sampling_policy),
        num_generate=len(local_array),
        local_array=local_array,
        speaker_nums=[0] * len(local_array),
    )

    print("generate", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--model_config_path", type=Path)
    parser.add_argument("--time_length", type=float, default=1)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--sampling_policy", default=SamplingPolicy.random.value)
    parser.add_argument("--use_gpu", action="store_true")
    speed_check(**vars(parser.parse_args()))
