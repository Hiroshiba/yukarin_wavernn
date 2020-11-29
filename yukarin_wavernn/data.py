import numpy


def encode_16bit(wave: numpy.ndarray):
    encoded = ((wave + 1) * 2 ** 15).astype(numpy.int64)
    encoded[encoded == 2 ** 16] = 2 ** 16 - 1
    coarse = encoded // 256
    fine = encoded % 256
    return coarse, fine


def encode_single(wave: numpy.ndarray, bit: int = 8):
    if 8 < bit:
        wave = wave.astype(numpy.float64)

    coarse = ((wave + 1) * 2 ** (bit - 1)).astype(numpy.int64)
    coarse[coarse == 2 ** bit] = 2 ** bit - 1
    return coarse


def decode_16bit(coarse, fine):
    signal = (coarse * 256 + fine) / (2 ** 16 - 1) * 2 - 1
    return signal.astype(numpy.float32) if hasattr(signal, "astype") else signal


def decode_single(coarse, bit: int = 8):
    signal = coarse / (2 ** bit - 1) * 2 - 1
    return signal.astype(numpy.float32) if hasattr(signal, "astype") else signal


def encode_mulaw(x: numpy.ndarray, mu: int):
    mu = mu - 1
    y = numpy.sign(x) * numpy.log1p(mu * numpy.abs(x)) / numpy.log1p(mu)
    return y


def decode_mulaw(x: numpy.ndarray, mu: int):
    mu = mu - 1
    y = numpy.sign(x) * ((1 + mu) ** numpy.abs(x) - 1) / mu
    return y
