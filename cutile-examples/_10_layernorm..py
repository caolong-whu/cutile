import torch
import cuda.tile as ct

@ct.kernel
def layernorm(
    x: ct.Array, o: ct.Array,
    w: ct.Array, b: ct.Array,
    eps: float, tile_size: ct.Constant,
):
    block_x = ct.bid(0)
    tile_x = ct.load(x, (block_x, 0), (1, tile_size))
    tile_x = ct.astype(tile_x, ct.float32)
    
    mean = ct.sum(tile_x) / tile_size
    tile_x = tile_x - mean
    
    tile_x_rsqrt = ct.rsqrt(ct.sum(tile_x  * tile_x) / tile_size + eps)
    tile_x = tile_x * tile_x_rsqrt
    
    tile_w = ct.load(w, (0, ), (tile_size, )).astype(ct.float32).reshape((1, tile_size))
    tile_b = ct.load(b, (0, ), (tile_size, )).astype(ct.float32).reshape((1, tile_size))
    tile_x = tile_w * tile_x + tile_b
    
    ct.store(o, (block_x, 0), tile_x.astype(o.dtype))
    
M, N = 16384, 1024

x = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
w = torch.randn([N, ], device="cuda", dtype=torch.bfloat16)
b = torch.randn([N, ], device="cuda", dtype=torch.bfloat16)
o = torch.empty_like(x)

real = torch.layer_norm(x, [N, ], w, b, 1e-7)

ct.launch(torch.cuda.current_stream(), (M, ), layernorm, (x, o, w, b, 1e-7, N))

print(real - o)