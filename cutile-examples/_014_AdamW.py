import torch
import cuda.tile as ct

@ct.kernel
def ct_fused_adamw(
    w: ct.Array, g: ct.Array, m: ct.Array, v: ct.Array,
    lr: float, step: int, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8, weight_decay: float=0.001,
    tile_size: ct.Constant=1024, allow_tma: ct.Constant=False
):
    block_x = ct.bid(0)
    
    tile_g = ct.load(g, (block_x,), (tile_size,), padding_mode=ct.PaddingMode.ZERO, allow_tma=allow_tma).astype(ct.float32)
    tile_m = ct.load(m, (block_x,), (tile_size,), padding_mode=ct.PaddingMode.ZERO, allow_tma=allow_tma).astype(ct.float32)
    tile_v = ct.load(v, (block_x,), (tile_size,), padding_mode=ct.PaddingMode.ZERO, allow_tma=allow_tma).astype(ct.float32)
    
    tile_m = beta1 * tile_m + (1.0 - beta1) * tile_g
    tile_v = beta2 * tile_v + (1.0 - beta2) * ct.pow(tile_g, 2)
    
    ct.store(m, (block_x,), tile_m.astype(m.dtype))
    ct.store(v, (block_x,), tile_v.astype(v.dtype))
    
    term_m = 1.0 / (1.0 - ct.pow(beta1, step))
    term_v = 1.0 / (1.0 - ct.pow(beta2, step))
    
    tile_m = tile_m * term_m
    tile_v = tile_v * term_v
    
    tile_w = ct.load(w, (block_x,), (tile_size,), padding_mode=ct.PaddingMode.ZERO, allow_tma=allow_tma).astype(ct.float32)
    tile_w = tile_w - lr * (tile_m / (ct.sqrt(tile_v) + eps) + weight_decay * tile_w)
    ct.store(w, (block_x,), tile_w.astype(w.dtype))
    
M = 16384
tile_size = 1024
w = torch.randn([M, ], device="cuda", dtype=torch.bfloat16)
g = torch.randn_like(w)
m = torch.zeros_like(w)
v = torch.zeros_like(w)
lr = 0.001
step = 1
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
weight_decay = 0.001
ct.launch(
    torch.cuda.current_stream(), (ct.cdiv(M, tile_size),),
    ct_fused_adamw, (w, g, m, v, lr, step, beta1, beta2, eps, weight_decay, tile_size, True)
)

print(w)