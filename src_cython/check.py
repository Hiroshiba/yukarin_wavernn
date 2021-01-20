import numpy
import torch
import yukarin_autoreg_cpp
from torch import Tensor
from yukarin_wavernn.config import NetworkConfig
from yukarin_wavernn.model import create_predictor
from yukarin_wavernn.network.fast_forward import fast_generate, get_fast_forward_params

max_batch_size = 4
graph_length = 1000
length = 4000
config = NetworkConfig(
    bit_size=10,
    hidden_size=896,
    local_size=5,
    conditioning_size=80,
    embedding_size=16,
    linear_hidden_size=896,
    local_scale=1,
    local_layer_num=1,
    speaker_size=0,
    speaker_embedding_size=0,
)
model = create_predictor(config).eval()
model.to("cuda")

local_size = config.conditioning_size * 2 if config.conditioning_size is not None else 0

base_x = torch.from_numpy(
    numpy.random.randint(0, config.bit_size ** 2, size=(max_batch_size)).astype(
        numpy.int32
    )
).to("cuda")
base_l_array = torch.from_numpy(
    numpy.random.rand(length, max_batch_size, local_size).astype(numpy.float32)
).to("cuda")
base_hidden = torch.from_numpy(
    numpy.random.rand(max_batch_size, config.hidden_size).astype(numpy.float32)
).to("cuda")

params = get_fast_forward_params(model)


def to_numpy(a):
    if isinstance(a, Tensor):
        a = a.detach().cpu().numpy()
    return numpy.ascontiguousarray(a)


# C++
yukarin_autoreg_cpp.initialize(
    graph_length=graph_length,
    max_batch_size=max_batch_size,
    local_size=local_size,
    hidden_size=config.hidden_size,
    embedding_size=config.embedding_size,
    linear_hidden_size=config.linear_hidden_size,
    output_size=2 ** config.bit_size,
    x_embedder_W=to_numpy(params["x_embedder_W"]),
    gru_xw=to_numpy(params["gru_xw"]),
    gru_xb=to_numpy(params["gru_xb"]),
    gru_hw=to_numpy(params["gru_hw"]),
    gru_hb=to_numpy(params["gru_hb"]),
    O1_W=to_numpy(params["O1_W"]),
    O1_b=to_numpy(params["O1_b"]),
    O2_W=to_numpy(params["O2_W"]),
    O2_b=to_numpy(params["O2_b"]),
)

before_output = None
for batch_size in [1, 2, 4]:
    x = base_x[:batch_size].clone()
    l_array = base_l_array[:, :batch_size].clone()
    hidden = base_hidden[:batch_size].clone()

    # x = model.xp.zeros_like(x)
    # l_array = model.xp.zeros_like(l_array)
    # hidden = model.xp.zeros_like(hidden)

    output = numpy.ones((length, batch_size), dtype=numpy.int32) * -1
    r = yukarin_autoreg_cpp.inference(
        batch_size=batch_size,
        length=length,
        output=output,
        x=to_numpy(x),
        l_array=to_numpy(l_array),
        hidden=to_numpy(hidden),
    )
    print(output)

    if before_output is not None:
        min_batch_size = min(before_output.shape[1], output.shape[1])
        flag = numpy.all(
            before_output[:, :min_batch_size] == output[:, :min_batch_size]
        )
        print("before_output == output :", flag)
    before_output = output


with torch.no_grad():
    expected = torch.stack(
        fast_generate(
            length=length,
            x=base_x.clone().to(torch.int64),
            l_array=torch.transpose(base_l_array.clone(), 0, 1),
            h=base_hidden.clone(),
            **params,
        )
    )
    print("expected", expected)

flag = numpy.all(output == to_numpy(expected))
print("output == expected :", flag)
