import unittest

import numpy
import torch
from parameterized import parameterized
from yukarin_wavernn.evaluator import GenerateEvaluator
from yukarin_wavernn.generator import Generator, SamplingPolicy

from tests.utility import get_test_config, get_test_model_path

bit = 10
mulaw = True
iteration = 3000


class TestEvaluator(unittest.TestCase):
    @parameterized.expand(
        [
            (4, 4),
        ]
    )
    def test_generate_evaluator(self, speaker_size, num_generate):
        config = get_test_config(
            bit=bit,
            mulaw=mulaw,
            speaker_size=speaker_size,
        )

        generator = Generator(
            config=config,
            predictor=get_test_model_path(
                bit=bit,
                mulaw=mulaw,
                speaker_size=speaker_size,
                iteration=iteration,
            ),
            use_gpu=True,
            use_fast_inference=False,
        )

        speaker_nums = list(range(num_generate)) if speaker_size > 0 else None
        waves = generator.generate(
            time_length=1,
            sampling_policy=SamplingPolicy.maximum,
            num_generate=num_generate,
            speaker_nums=speaker_nums,
        )

        wave = numpy.array([w.wave for w in waves])
        assert numpy.var(wave) > 0.2

        evaluator = GenerateEvaluator(
            generator=generator,
            time_length=1,
            local_padding_time_length=0,
            sampling_policy=SamplingPolicy.maximum,
        )
        scores = evaluator(
            wave=torch.from_numpy(wave),
            local=None,
            speaker_num=speaker_nums,
        )
        assert scores["mcd"][0] < 1
