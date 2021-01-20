import numpy
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from yukarin_wavernn.network.wave_rnn import WaveRNN


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
        x_embedder_W=x_embedder_W,
        gru_xw=gru_xw,
        gru_hw=gru_hw,
        gru_xb=gru_xb,
        gru_hb=gru_hb,
        O1_W=O1_W,
        O1_b=O1_b,
        O2_W=O2_W,
        O2_b=O2_b,
    )


def calc_gru_r(W_r_x, U_r_h):
    # r = torch.tanh((W_r_x + U_r_h) * half) * half + half
    r = W_r_x
    r += U_r_h
    r *= 0.5
    torch.tanh(r, out=r)
    r *= 0.5
    r += 0.5
    return r


def calc_gru_z(W_z_x, U_z_h):
    # z = torch.tanh((W_z_x + U_z_h) * half) * half + half
    z = W_z_x
    z += U_z_h
    z *= 0.5
    torch.tanh(z, out=z)
    z *= 0.5
    z += 0.5
    return z


def calc_gru_h_bar(r, U_x, W_x):
    # h_bar = torch.tanh(W_x + r * U_x)
    r *= U_x
    r += W_x
    torch.tanh(r, out=r)
    return r


def calc_gru_hidden(hidden, z, h_bar):
    # new_hidden = z * hidden + (one - z) * h_bar
    hidden *= z
    z *= -1
    z += 1
    h_bar *= z
    hidden += h_bar
    return hidden


def gru_element_wise(hidden, W_r_x, W_z_x, W_x, U_r_h, U_z_h, U_x):
    r = calc_gru_r(W_r_x, U_r_h)
    z = calc_gru_z(W_z_x, U_z_h)
    h_bar = calc_gru_h_bar(r, U_x, W_x)
    return calc_gru_hidden(hidden, z, h_bar)


def fast_forward_one(
    prev_x: Tensor,
    prev_l: Tensor,
    hidden: Tensor,
    x_embedder_W: Tensor,
    gru_xw: Tensor,
    gru_hw: Tensor,
    gru_xb: Tensor,
    gru_hb: Tensor,
    O1_W: Tensor,
    O1_b: Tensor,
    O2_W: Tensor,
    O2_b: Tensor,
    w_gru_x: Tensor,
    w_gru_h: Tensor,
    w_out_x1: Tensor,
    w_out_x2: Tensor,
):
    prev_xl = torch.cat((x_embedder_W[prev_x], prev_l), dim=1)  # (batch_size, ?)

    # gru_x = prev_xl.dot(gru_xw) + gru_xb
    gru_x = w_gru_x
    torch.mm(prev_xl, gru_xw, out=gru_x)
    gru_x += gru_xb

    # gru_h = hidden.dot(gru_hw) + gru_hb
    gru_h = w_gru_h
    torch.mm(hidden, gru_hw, out=gru_h)
    gru_h += gru_hb

    size = gru_x.shape[1] // 3
    W_r_x, W_z_x, W_x = gru_x[:, :size], gru_x[:, size : size * 2], gru_x[:, size * 2 :]
    U_r_h, U_z_h, U_x = gru_h[:, :size], gru_h[:, size : size * 2], gru_h[:, size * 2 :]
    new_hidden = gru_element_wise(hidden, W_r_x, W_z_x, W_x, U_r_h, U_z_h, U_x)

    # out_x = new_hidden.dot(O1_W) + O1_b
    out_x1 = w_out_x1
    torch.mm(new_hidden, O1_W, out=out_x1)
    out_x1 += O1_b

    torch.maximum(out_x1, torch.tensor(0.0).to(out_x1.device), out=out_x1)

    # out_x = out_x.dot(O2_W) + O2_b
    out_x2 = w_out_x2
    torch.mm(out_x1, O2_W, out=out_x2)
    out_x2 += O2_b
    return out_x2, new_hidden


def fast_generate(
    length: int,
    x: Tensor,
    l_array: Tensor,
    h: Tensor,
    x_embedder_W: Tensor,
    gru_xw: Tensor,
    gru_hw: Tensor,
    gru_xb: Tensor,
    gru_hb: Tensor,
    O1_W: Tensor,
    O1_b: Tensor,
    O2_W: Tensor,
    O2_b: Tensor,
):
    batchsize = len(x)
    w_gru_x = torch.empty((batchsize, len(gru_xb)), dtype=h.dtype, device=x.device)
    w_gru_h = torch.empty((batchsize, len(gru_hb)), dtype=h.dtype, device=x.device)
    w_out_x1 = torch.empty((batchsize, len(O1_b)), dtype=h.dtype, device=x.device)
    w_out_x2 = torch.empty((batchsize, len(O2_b)), dtype=h.dtype, device=x.device)

    output = []
    for i in tqdm(range(length), desc="fast_generate"):
        dist, h = fast_forward_one(
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

        # softmax
        dist = F.log_softmax(dist.double(), dim=1)

        # sampling
        random = torch.from_numpy(numpy.random.gumbel(size=dist.shape)).to(dist.device)
        x = (dist + random).argmax(dim=1)
        output.append(x)

    return output
