import torch
import cuda.tile as ct
def rope_original(x: torch.Tensor, theta: int = 10000):
    # x: [batch_size, num_heads, seq_len, head_dim]
    batch_size, num_heads, seq_len, head_dim = x.shape

    # freqs = 10000 ^ (-2i / head_dim), i = 0, 1, ..., head_dim // 2 - 1
    freqs = torch.pow(theta, -torch.arange(0, head_dim, 2, dtype=torch.float, device=x.device) / head_dim)

    # pos: [seq_len, 1]
    pos = torch.arange(0, seq_len, dtype=torch.float, device=x.device)

    # pos_embed: [seq_len, head_dim // 2]
    pos_embed = torch.outer(pos, freqs)

    cos_embed = torch.cos(pos_embed)
    sin_embed = torch.sin(pos_embed)

    # cos_embed: [seq_len, head_dim // 2] -> [seq_len, head_dim]
    cos_embed = cos_embed.repeat_interleave(2, dim=-1)
    # sin_embed: [seq_len, head_dim // 2] -> [seq_len, head_dim]
    sin_embed = sin_embed.repeat_interleave(2, dim=-1)
    
    # x2: [batch_size, num_heads, seq_len, head_dim // 2, 2]
    x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
    # x2: [batch_size, num_heads, seq_len, head_dim]
    x2 = x2.flatten(-2)

    x = x * cos_embed + x2 * sin_embed

    return x

def rope_llama(x: torch.Tensor, theta: int = 10000):
    # x: [batch_size, num_heads, seq_len, head_dim]
    batch_size, num_heads, seq_len, head_dim = x.shape

    # freqs = 10000 ^ (-2i / head_dim), i = 0, 1, ..., head_dim // 2 - 1
    freqs = torch.pow(theta, -torch.arange(0, head_dim, 2, dtype=torch.float, device=x.device) / head_dim)

    # pos: [seq_len, 1]
    pos = torch.arange(0, seq_len, dtype=torch.float)

    # pos_embed: [seq_len, head_dim // 2]
    pos_embed = torch.outer(pos, freqs)

    cos_embed = torch.cos(pos_embed)
    sin_embed = torch.sin(pos_embed)

    cos_embed = torch.cat([cos_embed, cos_embed], dim=-1)
    sin_embed = torch.cat([sin_embed, sin_embed], dim=-1)

    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    x_half_rotated = torch.cat([-x2, x1], dim=-1)

    x = x * cos_embed + x_half_rotated * sin_embed

    return x

def rope_complex(x: torch.Tensor, theta: int = 10000):
    # x: [batch_size, num_heads, seq_len, head_dim]
    batch_size, num_heads, seq_len, head_dim = x.shape

    freqs = torch.pow(theta, -torch.arange(0, head_dim, 2) / head_dim, dtype=torch.float, device=x.device)

    pos = torch.arange(0, seq_len, dtype=torch.float, device=x.device)

    pos_embed = torch.outer(pos, freqs)

    # e ^ (i * pos_embed)
    pos_embed = torch.polar(torch.ones_like(pos_embed), pos_embed)

    # x: [batch_size, num_heads, seq_len, head_dim // 2, 2] -> [batch_size, num_heads, seq_len, head_dim // 2]
    x_complex = torch.view_as_complex(x.float().view(batch_size, num_heads, seq_len, head_dim // 2, -1))

    x_rotated = x_complex * pos_embed

    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out 

@ct.kernel
def build_freqs(o: ct.Array, head_dim: int, theta: float, 
                tile_size: ct.Constant, allow_tma: ct.Constant):
    
    # o: [max_seq_len, head_dim // 2, 2, 2]
    # block_x -> head_dim // 2
    # block_y -> max_seq_len
    block_x, block_y = ct.bid(0), ct.bid(1)
    # tile_o: [tile_size]
    tile_o = block_x * tile_size + ct.arange(tile_size, dtype=torch.float32)
    # angle = theta ^ (-2 * i) / d
    tile_o = 1.0 / ct.pow(theta, tile_o / (head_dim / 2))
    # pos * angle
    tile_o = block_y * tile_o

    sin = ct.sin(tile_o).reshape((1, tile_size, 1, 1))
    cos = ct.cos(tile_o).reshape((1, tile_size, 1, 1))
    r1 = ct.cat((cos, -sin), axis=-1)
    r2 = ct.cat((sin, cos), axis=-1)
    rotation_matrix = ct.cat((r1, r2), axis=-2)

    ct.store(o, (block_y, block_x, 0, 0), rotation_matrix.astype(o.dtype), allow_tma=allow_tma)

@ct.kernel
def apply_rope(x: ct.Array, coord: ct.Array, o: ct.Array, freqs: ct.Array, tile_size: ct.Constant):
    """
    x: [seq_len, head_dim]
    coord: [max_seq_len]
    o: [seq_len, head_dim]
    freqs: [max_seq_len, head_dim // 2, 2, 2]
    """
    block_x, block_y = ct.bid(0), ct.bid(1)
    pos = ct.load(coord, (block_y, ), (1, ), allow_tma=False)
    tile_x = ct.load(x, (block_y, block_x), (1, tile_size), allow_tma=False).astype(ct.float32)
    tile_f = ct.load(freqs, (pos.item(), block_x, 0, 0), (1, tile_size // 2, 2, 2), allow_tma=False).astype(ct.float32)
    
    # To apply R * x, [2, 2] * [2, 1] -> [2, 1]
    tile_x = tile_x.reshape((-1, 2, 1))
    tile_f = tile_f.reshape((-1, 2, 2))

    tile_o = ct.full(shape=(tile_x.shape), fill_value=0.0, dtype=ct.float32)
    tile_o = ct.mma(tile_f, tile_x, tile_o).reshape((1, tile_size))
    ct.store(o, (block_y, block_x), tile_o.astype(o.dtype), allow_tma=False)

def get_freqs(max_seq_len: int, hidde_dim: int, theta: float = 10000.0, tile_size: int = 32):
    
    freqs = torch.empty([max_seq_len, hidde_dim, 2, 2], device="cuda", dtype=torch.bfloat16)
    grid = (ct.cdiv(hidde_dim, tile_size), max_seq_len)
    ct.launch(torch.cuda.current_stream(), grid, build_freqs, (freqs, hidde_dim, theta, tile_size, False))

    return freqs

def apply_rope_cutile(x: torch.Tensor, coord: torch.Tensor, freqs: torch.Tensor, tile_size):

    seq_len, hidden_dim = x.shape

    o = torch.empty_like(x)
    grid = (ct.cdiv(hidden_dim, tile_size), seq_len)
    ct.launch(torch.cuda.current_stream(), grid, apply_rope, (x, coord, o, freqs, tile_size))

    return o

if __name__ == "__main__":

    MAX_SEQ_LEN = 128
    SEQ_LEN = 16
    HIDDEN_DIM = 128
    TILE_SIZE = 32

    x = torch.randn([SEQ_LEN, HIDDEN_DIM], device="cuda", dtype=torch.bfloat16)
    x_embed = torch.empty_like(x)
    freqs = get_freqs(MAX_SEQ_LEN, HIDDEN_DIM, 10000.0, TILE_SIZE)

    coord = torch.arange(MAX_SEQ_LEN, device="cuda", dtype=torch.int32)
    
    x_embed = apply_rope_cutile(x, coord, freqs, TILE_SIZE)

    real = rope_original(x, 10000.0)
    print(x_embed - real)


