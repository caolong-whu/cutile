import torch
import cuda.tile as ct

@ct.kernel
def silu_fuse_mul(x: ct.Array, gate: ct.Array, o: ct.Array, tile_size: ct.Constant):
    
    block_x = ct.bid(0)
    tile_x = ct.load(x, (block_x, ), (tile_size, ), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    tile_gate = ct.load(gate, (block_x, ), (tile_size, ), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)

    tile_o = tile_x * (1 / (1 + ct.exp(-tile_gate)))

    ct.store(o, (block_x, ), tile_o.astype(o.dtype))

M, N = 128, 1024
x = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
gate = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
o = torch.empty_like(x)
ct.launch(torch.cuda.current_stream(), (ct.cdiv(M * N, 1024), ), silu_fuse_mul, (x.flatten(), gate.flatten(), o.flatten(), 1024))

print(o)
