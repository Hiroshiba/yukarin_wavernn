import unittest
from pathlib import Path

from parameterized import parameterized
from yukarin_wavernn.generator import Generator, SamplingPolicy

from tests.utility import get_test_config, get_test_model_path

bit = 10
mulaw = True
iteration = 3000


class TestGenerator(unittest.TestCase):
    @parameterized.expand(
        [
            (0, 1),
            (4, 1),
            (4, 4),
        ]
    )
    def test_generator(self, speaker_size, num_generate):
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
        )

        for sampling_policy in SamplingPolicy.__members__.values():
            with self.subTest(sampling_policy=sampling_policy):
                waves = generator.generate(
                    time_length=0.1,
                    sampling_policy=sampling_policy,
                    num_generate=num_generate,
                    speaker_nums=(
                        list(range(num_generate)) if speaker_size > 0 else None
                    ),
                )
                for num, wave in enumerate(waves):
                    wave.save(
                        Path(
                            "/tmp/"
                            f"test_generator_audio"
                            f"-sampling_policy={sampling_policy}"
                            f"-bit={bit}"
                            f"-mulaw={mulaw}"
                            f"-speaker_size={speaker_size}"
                            f"-num={num}"
                            f"-iteration={iteration}"
                            ".wav"
                        )
                    )
