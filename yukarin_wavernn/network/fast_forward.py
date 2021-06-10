import numpy
from numba import njit
from torch.tensor import Tensor
from yukarin_wavernn.network.wave_rnn import WaveRNN


def to_numpy(a):
    if isinstance(a, Tensor):
        a = a.detach().cpu().numpy()
    return numpy.ascontiguousarray(a)


def get_fast_forward_params(model: WaveRNN):
    x_embedder_W = model.x_embedder.weight

    gru_xw = model.gru.weight_ih_l0.T
    gru_hw = model.gru.weight_hh_l0.T
    gru_xb = model.gru.bias_ih_l0
    gru_hb = model.gru.bias_hh_l0

    O1_W = model.O1.weight.T
    O1_b = model.O1.bias
    O2_W = model.O2.weight.T
    O2_b = model.O2.bias

    return dict(
        x_embedder_W=to_numpy(x_embedder_W),
        gru_xw=to_numpy(gru_xw),
        gru_hw=to_numpy(gru_hw),
        gru_xb=to_numpy(gru_xb),
        gru_hb=to_numpy(gru_hb),
        O1_W=to_numpy(O1_W),
        O1_b=to_numpy(O1_b),
        O2_W=to_numpy(O2_W),
        O2_b=to_numpy(O2_b),
    )


@njit(nogil=True)
def calc_gru_r(W_r_x: numpy.ndarray, U_r_h: numpy.ndarray):
    # r = numpy.tanh((W_r_x +: numpy.ndarray U_r_h) * half) * half + half
    r = W_r_x
    r += U_r_h
    r *= 0.5
    numpy.tanh(r, r)
    r *= 0.5
    r += 0.5
    return r


@njit(nogil=True)
def calc_gru_z(W_z_x: numpy.ndarray, U_z_h: numpy.ndarray):
    # z = numpy.tanh((W_z_x + U_z_h) * half) * half + half
    z = W_z_x
    z += U_z_h
    z *= 0.5
    numpy.tanh(z, z)
    z *= 0.5
    z += 0.5
    return z


@njit(nogil=True)
def calc_gru_h_bar(r: numpy.ndarray, U_x: numpy.ndarray, W_x: numpy.ndarray):
    # h_bar = numpy.tanh(W_x + r * U_x)
    r *= U_x
    r += W_x
    numpy.tanh(r, r)
    return r


@njit(nogil=True)
def calc_gru_hidden(hidden: numpy.ndarray, z: numpy.ndarray, h_bar: numpy.ndarray):
    # new_hidden = z * hidden + (one - z) * h_bar
    hidden *= z
    z *= -1
    z += 1
    h_bar *= z
    hidden += h_bar
    return hidden


@njit(nogil=True)
def gru_element_wise(
    hidden: numpy.ndarray,
    W_r_x: numpy.ndarray,
    W_z_x: numpy.ndarray,
    W_x: numpy.ndarray,
    U_r_h: numpy.ndarray,
    U_z_h: numpy.ndarray,
    U_x: numpy.ndarray,
):
    r = calc_gru_r(W_r_x, U_r_h)
    z = calc_gru_z(W_z_x, U_z_h)
    h_bar = calc_gru_h_bar(r, U_x, W_x)
    return calc_gru_hidden(hidden, z, h_bar)


@njit(nogil=True)
def fast_forward_one(
    prev_x: numpy.ndarray,
    prev_l: numpy.ndarray,
    hidden: numpy.ndarray,
    x_embedder_W: numpy.ndarray,
    gru_xw: numpy.ndarray,
    gru_hw: numpy.ndarray,
    gru_xb: numpy.ndarray,
    gru_hb: numpy.ndarray,
    O1_W: numpy.ndarray,
    O1_b: numpy.ndarray,
    O2_W: numpy.ndarray,
    O2_b: numpy.ndarray,
    w_gru_x: numpy.ndarray,
    w_gru_h: numpy.ndarray,
    w_out_x1: numpy.ndarray,
    w_out_x2: numpy.ndarray,
):
    prev_xl = numpy.concatenate(
        (x_embedder_W[prev_x], prev_l), axis=1
    )  # (batch_size, ?)

    # gru_x = prev_xl.dot(gru_xw) + gru_xb
    gru_x = w_gru_x
    numpy.dot(prev_xl, gru_xw, gru_x)
    gru_x += gru_xb

    # gru_h = hidden.dot(gru_hw) + gru_hb
    gru_h = w_gru_h
    numpy.dot(hidden, gru_hw, gru_h)
    gru_h += gru_hb

    size = gru_x.shape[1] // 3
    W_r_x, W_z_x, W_x = gru_x[:, :size], gru_x[:, size : size * 2], gru_x[:, size * 2 :]
    U_r_h, U_z_h, U_x = gru_h[:, :size], gru_h[:, size : size * 2], gru_h[:, size * 2 :]
    new_hidden = gru_element_wise(hidden, W_r_x, W_z_x, W_x, U_r_h, U_z_h, U_x)

    # out_x = new_hidden.dot(O1_W) + O1_b
    out_x1 = w_out_x1
    numpy.dot(new_hidden, O1_W, out_x1)
    out_x1 += O1_b

    numpy.maximum(out_x1, 0, out_x1)

    # out_x = out_x.dot(O2_W) + O2_b
    out_x2 = w_out_x2
    numpy.dot(out_x1, O2_W, out_x2)
    out_x2 += O2_b
    return out_x2, new_hidden


@njit(nogil=True)
def _max_axis1_keepdims(array: numpy.ndarray):
    out = numpy.empty((array.shape[0], 1), dtype=array.dtype)
    for i in range(array.shape[0]):
        out[i] = array[i].max()
    return out


@njit(nogil=True)
def _random_choice_p(prob: numpy.ndarray):
    cumsum = numpy.cumsum(prob)
    rand = numpy.random.random() * cumsum[-1]
    return numpy.searchsorted(cumsum, rand, side="right")


@njit(nogil=True)
def fast_generate(
    length: int,
    x: numpy.ndarray,
    l_array: numpy.ndarray,
    h: numpy.ndarray,
    x_embedder_W: numpy.ndarray,
    gru_xw: numpy.ndarray,
    gru_hw: numpy.ndarray,
    gru_xb: numpy.ndarray,
    gru_hb: numpy.ndarray,
    O1_W: numpy.ndarray,
    O1_b: numpy.ndarray,
    O2_W: numpy.ndarray,
    O2_b: numpy.ndarray,
    low_probability_threshold: float,
):
    batchsize = len(x)
    w_gru_x = numpy.empty((batchsize, len(gru_xb)), dtype=h.dtype)
    w_gru_h = numpy.empty((batchsize, len(gru_hb)), dtype=h.dtype)
    w_out_x1 = numpy.empty((batchsize, len(O1_b)), dtype=h.dtype)
    w_out_x2 = numpy.empty((batchsize, len(O2_b)), dtype=h.dtype)

    output = numpy.empty((length, batchsize), dtype=x.dtype)
    for i in range(length):
        d, h = fast_forward_one(
            prev_x=x,
            prev_l=l_array[:, i],
            hidden=h,
            x_embedder_W=x_embedder_W,
            gru_xw=gru_xw,
            gru_hw=gru_hw,
            gru_xb=gru_xb,
            gru_hb=gru_hb,
            O1_W=O1_W,
            O1_b=O1_b,
            O2_W=O2_W,
            O2_b=O2_b,
            w_gru_x=w_gru_x,
            w_gru_h=w_gru_h,
            w_out_x1=w_out_x1,
            w_out_x2=w_out_x2,
        )
        dist = d.astype(numpy.float64)
        dist[dist < low_probability_threshold] -= 200

        dist -= _max_axis1_keepdims(dist)
        numpy.exp(dist, dist)
        dist /= _max_axis1_keepdims(dist)

        for j in range(batchsize):
            x[j] = _random_choice_p(dist[j])

        output[i] = x

    return output
