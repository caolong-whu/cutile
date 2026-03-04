import torch
import cuda.tile as ct


@ct.function
def layernorm(
    tile_x: ct.Tile,
    tile_w: ct.Tile,
    tile_b: ct.Tile,
    tile_size: ct.Constant,
    normalize_dim: int,
    eps: float,
):
    tile_x = tile_x.astype(ct.float32)
    tile_w = tile_w.astype(ct.float32)
    tile_b = tile_b.astype(ct.float32)

    # only work when tile_size > normalization_dim
    mask = ct.arange(tile_size, dtype=ct.int32) < normalize_dim

    inv_normalization_dim = 1 / normalize_dim

    mean = ct.sum(tile_x) * inv_normalization_dim
    tile_x = tile_x - mean  # [..., -dim, -dim, -dim, ...]
    tile_x = tile_x * mask  # [..., 0, 0, 0, ...]

    rsqrt = ct.rsqrt(ct.sum(tile_x * tile_x) * inv_normalization_dim + eps)
    tile_x = tile_x * rsqrt

    return tile_x * tile_w + tile_b


@ct.kernel
def _AdaLayerNorm(
    x: ct.Array,
    w: ct.Array,
    b: ct.Array,
    shift: ct.Array,
    scale: ct.Array,
    eps: float,
    o: ct.Array,
    allow_tma: ct.Constant,
    normalize_dim: int,
    tile_size: ct.Constant,
):
    block_b, block_s, block_x = ct.bid(2), ct.bid(1), ct.bid(0)

    tile_x = ct.load(
        x,
        (block_b, block_s, block_x),
        (1, 1, tile_size),
        allow_tma=allow_tma,
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_x = tile_x.reshape((tile_size,))

    tile_w = ct.load(
        w,
        (block_x,),
        (tile_size,),
        allow_tma=allow_tma,
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_b = ct.load(
        b,
        (block_x,),
        (tile_size,),
        allow_tma=allow_tma,
        padding_mode=ct.PaddingMode.ZERO,
    )

    tile_x = layernorm(tile_x, tile_w, tile_b, tile_size, normalize_dim, eps)

    tile_shift = ct.load(
        shift,
        (block_b, block_x),
        (1, tile_size),
        allow_tma=False,
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_scale = ct.load(
        scale,
        (block_b, block_x),
        (1, tile_size),
        allow_tma=False,
        padding_mode=ct.PaddingMode.ZERO,
    )

    tile_shift = tile_shift.reshape((tile_size,)).astype(ct.float32)
    tile_scale = tile_scale.reshape((tile_size,)).astype(ct.float32)

    tile_x = tile_x * (1 + tile_scale) + tile_shift
    tile_x = tile_x.reshape((1, 1, tile_size))
    ct.store(
        o, (block_b, block_s, block_x), tile_x.astype(o.dtype), allow_tma=allow_tma
    )


def AdaLayerNorm(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:

    batch, seqlen, dim = x.shape
    tile_size = 4096

    o = torch.empty_like(x)
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(dim, tile_size), seqlen, batch),
        _AdaLayerNorm,
        (x, w, b, shift, scale, eps, o, False, dim, tile_size),
    )

    return o


if __name__ == "__main__":
    batch, seqlen, dim = 1, 49600, 3072
    x = torch.rand([batch, seqlen, dim], dtype=torch.float32, device="cuda")
    w = torch.rand(
        [
            dim,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    b = torch.rand(
        [
            dim,
        ],
        dtype=torch.float32,
        device="cuda",
    )

    shift = torch.rand([batch, dim], dtype=torch.float32, device="cuda")
    scale = torch.rand([batch, dim], dtype=torch.float32, device="cuda")

    ref = torch.nn.functional.layer_norm(x, (dim,), w, b, eps=1e-7)
    ref = ref * (1 + scale[:, None]) + shift[:, None]

    ct_result = AdaLayerNorm(x, w, b, shift, scale, 1e-7)

    print(ref - ct_result)
