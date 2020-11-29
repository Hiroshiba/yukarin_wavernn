import unittest

import numpy
from yukarin_wavernn.data import (
    decode_16bit,
    decode_mulaw,
    decode_single,
    encode_16bit,
    encode_mulaw,
    encode_single,
)


class TestEncode16Bit(unittest.TestCase):
    def setUp(self):
        self.wave_increase = numpy.linspace(-1, 1, num=2 ** 16).astype(numpy.float32)

    def test_encode(self):
        coarse, fine = encode_16bit(self.wave_increase)

        self.assertTrue(numpy.all(coarse >= 0))
        self.assertTrue(numpy.all(coarse < 256))
        self.assertTrue(numpy.all(fine >= 0))
        self.assertTrue(numpy.all(fine < 256))

        self.assertTrue(numpy.all(numpy.diff(coarse) >= 0))
        self.assertTrue((numpy.diff(fine) >= 0).sum() == 256 ** 2 - 255 - 1)


class TestEncodeSingle(unittest.TestCase):
    def setUp(self):
        self.wave_increase = numpy.linspace(-1, 1, num=2 ** 16).astype(numpy.float64)

    def test_encode_single(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                coarse = encode_single(self.wave_increase, bit=bit)

                self.assertTrue(numpy.all(coarse >= 0))
                self.assertTrue(numpy.all(coarse < 2 ** bit))

                self.assertTrue(numpy.all(numpy.diff(coarse) >= 0))

                hist, _ = numpy.histogram(coarse, bins=2 ** bit, range=[0, 2 ** bit])
                numpy.testing.assert_equal(
                    hist, numpy.ones(2 ** bit) * (2 ** 16 / 2 ** bit)
                )

    def test_encode_single_float32(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                coarse = encode_single(
                    self.wave_increase.astype(numpy.float32), bit=bit
                )

                self.assertTrue(numpy.all(coarse >= 0))
                self.assertTrue(numpy.all(coarse < 2 ** bit))

                self.assertTrue(numpy.all(numpy.diff(coarse) >= 0))

                hist, _ = numpy.histogram(coarse, bins=2 ** bit, range=[0, 2 ** bit])
                all_equal = numpy.all(
                    hist == numpy.ones(2 ** bit) * (2 ** 16 / 2 ** bit)
                )
                if bit <= 10:
                    self.assertTrue(all_equal)
                else:
                    self.assertFalse(all_equal)


class TestDecode16Bit(unittest.TestCase):
    def setUp(self):
        self.wave_increase = numpy.linspace(-1, 1, num=2 ** 17).astype(numpy.float32)

    def test_decode(self):
        c, f = numpy.meshgrid(numpy.arange(256), numpy.arange(256))
        w = decode_16bit(c.ravel(), f.ravel())

        self.assertTrue(numpy.all(-1 <= w))
        self.assertTrue(numpy.all(w <= 1))

        self.assertEqual((w < 0).sum(), 2 ** 15)
        self.assertEqual((w >= 0).sum(), 2 ** 15)

    def test_encode_decode(self):
        coarse, fine = encode_16bit(self.wave_increase)
        w = decode_16bit(coarse, fine)

        numpy.testing.assert_allclose(self.wave_increase, w, atol=2 ** -15)


class TestDecodeSingle(unittest.TestCase):
    def test_decode(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                coarse = numpy.arange(2 ** bit).astype(numpy.int64)
                w = decode_single(coarse, bit=bit)
                numpy.testing.assert_equal(
                    w, numpy.linspace(-1, 1, num=2 ** bit).astype(numpy.float32)
                )

    def test_decode_one_value(self):
        self.assertEqual(decode_single(0), -1)
        self.assertEqual(decode_single(255), 1)


class TestMulaw(unittest.TestCase):
    def setUp(self):
        self.dummy_array = numpy.linspace(-1, 1, num=2 ** 16).astype(numpy.float32)

    def test_encode_mulaw(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                mu = 2 ** bit
                y = encode_mulaw(self.dummy_array, mu=mu)
                self.assertEqual(y.min(), -1)
                self.assertEqual(y.max(), 1)

    def test_decode_mulaw(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                mu = 2 ** bit
                y = decode_mulaw(self.dummy_array, mu=mu)
                self.assertEqual(y.min(), -1)
                self.assertEqual(y.max(), 1)

    def test_encode_decode_mulaw(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                mu = 2 ** bit
                x = encode_mulaw(self.dummy_array, mu=mu)
                y = decode_mulaw(x, mu=mu)
                numpy.testing.assert_allclose(self.dummy_array, y, atol=1 ** -mu)
