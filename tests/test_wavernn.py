import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized_class
from yukarin_wavernn.network.fast_forward import (
    fast_forward_one,
    get_fast_forward_params,
)
from yukarin_wavernn.network.wave_rnn import WaveRNN

batch_size = 2
length = 3
hidden_size = 8
loal_size = 5
bit_size = 9
speaker_size = 10


def _make_hidden():
    hidden = torch.rand(batch_size, hidden_size, dtype=torch.float32)
    return hidden


def get_fast_forward_params_one(wave_rnn):
    fast_forward_params = get_fast_forward_params(wave_rnn)
    dtype = fast_forward_params["gru_xb"].dtype
    fast_forward_params["w_gru_x"] = torch.empty(
        (batch_size, len(fast_forward_params["gru_xb"])), dtype=dtype
    )
    fast_forward_params["w_gru_h"] = torch.empty(
        (batch_size, len(fast_forward_params["gru_hb"])), dtype=dtype
    )
    fast_forward_params["w_out_x1"] = torch.empty(
        (batch_size, len(fast_forward_params["O1_b"])), dtype=dtype
    )
    fast_forward_params["w_out_x2"] = torch.empty(
        (batch_size, len(fast_forward_params["O2_b"])), dtype=dtype
    )
    return fast_forward_params


@parameterized_class(["with_speaker"], [[False], [True]])
class TestWaveRNN(unittest.TestCase):
    def setUp(self):
        with_speaker = self.with_speaker

        self.x_array = torch.randint(
            0, bit_size ** 2, size=[batch_size, length], dtype=torch.int64
        )
        self.x_one = torch.randint(
            0, bit_size ** 2, size=[batch_size, 1], dtype=torch.int64
        )

        self.l_array = torch.rand(batch_size, length, loal_size, dtype=torch.float32)
        self.l_one = torch.rand(batch_size, 1, loal_size, dtype=torch.float32)

        if with_speaker:
            self.s_one = torch.randint(
                0,
                speaker_size,
                size=[
                    batch_size,
                ],
                dtype=torch.int64,
            )
        else:
            self.s_one = None

        wave_rnn = WaveRNN(
            bit_size=bit_size,
            conditioning_size=7,
            embedding_size=32,
            hidden_size=hidden_size,
            linear_hidden_size=11,
            local_size=loal_size,
            local_scale=1,
            local_layer_num=2,
            speaker_size=speaker_size if with_speaker else 0,
            speaker_embedding_size=3 if with_speaker else 0,
        ).eval()

        self.wave_rnn = wave_rnn

    def test_call(self):
        wave_rnn = self.wave_rnn
        wave_rnn(
            x_array=self.x_array,
            l_array=self.l_array,
            s_one=self.s_one,
        )

    def test_call_with_local_padding(self):
        local_padding_size = 5

        wave_rnn = self.wave_rnn
        with self.assertRaises(Exception):
            wave_rnn(
                x_array=self.x_array,
                l_array=self.l_array,
                s_one=self.s_one,
                local_padding_size=local_padding_size,
            )

        l_array = F.pad(
            self.l_array,
            pad=(0, 0, local_padding_size, local_padding_size, 0, 0),
            mode="constant",
        )
        wave_rnn(
            x_array=self.x_array,
            l_array=l_array,
            s_one=self.s_one,
            local_padding_size=local_padding_size,
        )

    def test_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one)
        wave_rnn.forward_one(
            self.x_one[:, 0],
            l_one[:, 0],
            hidden=hidden,
        )

    def test_fast_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one)
        fast_forward_params = get_fast_forward_params_one(wave_rnn)
        with torch.no_grad():
            fast_forward_one(
                prev_x=self.x_one[:, 0],
                prev_l=l_one[:, 0],
                hidden=hidden,
                **fast_forward_params,
            )

    def test_batchsize1_forward(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one)

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array,
            l_array=l_array,
            hidden=hidden,
        )

        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:1],
            l_array=l_array[:1],
            hidden=hidden[:1],
        )

        assert torch.allclose(oa[:1], ob, atol=1e-6)
        assert torch.allclose(ha[:1], hb, atol=1e-6)

    def test_batchsize1_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one)

        oa, ha = wave_rnn.forward_one(
            self.x_one[:, 0],
            l_one[:, 0],
            hidden=hidden,
        )

        ob, hb = wave_rnn.forward_one(
            self.x_one[:1, 0],
            l_one[:1, 0],
            hidden=hidden[:1],
        )

        assert torch.allclose(oa[:1], ob, atol=1e-6)
        assert torch.allclose(ha[:1], hb, atol=1e-6)

    def test_batchsize1_fast_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one)
        fast_forward_params = get_fast_forward_params_one(wave_rnn)

        with torch.no_grad():
            oa, ha = fast_forward_one(
                prev_x=self.x_one[:, 0],
                prev_l=l_one[:, 0],
                hidden=hidden,
                **fast_forward_params,
            )

            ob, hb = fast_forward_one(
                prev_x=self.x_one[:1, 0],
                prev_l=l_one[:1, 0],
                hidden=hidden[:1],
                **fast_forward_params,
            )

        assert torch.allclose(oa[:1], ob, atol=1e-6)
        assert torch.allclose(ha[:1], hb, atol=1e-6)

    def test_same_call_and_forward(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()

        oa, ha = wave_rnn(
            x_array=self.x_array,
            l_array=self.l_array,
            s_one=self.s_one,
            hidden=hidden,
        )

        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one)
        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:, :-1],
            l_array=l_array[:, 1:],
            hidden=hidden,
        )

        assert torch.equal(oa, ob)
        assert torch.equal(ha, hb)

    def test_same_forward_rnn_and_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one)

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array,
            l_array=l_array,
            hidden=hidden,
        )

        hb = hidden
        for i, (x, l) in enumerate(
            zip(
                torch.split(self.x_array, 1, dim=1),
                torch.split(l_array, 1, dim=1),
            )
        ):
            ob, hb = wave_rnn.forward_one(
                x[:, 0],
                l[:, 0],
                hb,
            )

            assert torch.allclose(oa[:, :, i], ob, atol=1e-6)

        assert torch.allclose(ha, hb, atol=1e-6)

    def test_same_forward_rnn_and_fast_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one)
        fast_forward_params = get_fast_forward_params_one(wave_rnn)

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array,
            l_array=l_array,
            hidden=hidden,
        )

        hb = hidden
        for i, (x, l) in enumerate(
            zip(
                torch.split(self.x_array, 1, dim=1),
                torch.split(l_array, 1, dim=1),
            )
        ):
            with torch.no_grad():
                ob, hb = fast_forward_one(
                    prev_x=x[:, 0],
                    prev_l=l[:, 0],
                    hidden=hb,
                    **fast_forward_params,
                )

            assert torch.allclose(oa[:, :, i], ob, atol=1e-6)

        assert torch.allclose(ha, hb, atol=1e-6)
