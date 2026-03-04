import torch
import cuda.tile as ct


@ct.kernel
def symm_matmul(
    X: ct.Array, O: ct.Array, alpha: float, tileMN: ct.Constant, tileK: ct.Constant
):
    # O = alpha * X @ X.T
    # [M, M]
    block_x, block_y = ct.bid(0), ct.bid(1)

    if block_y > block_x:
        return

    num_tile_k = ct.cdiv(X.shape[-1], tileK)
    acc = ct.full((tileMN, tileMN), 0.0, dtype=ct.float32)

    for k in range(num_tile_k):
        tile_x = ct.load(X, (block_x, k), (tileMN, tileK))
        tile_t = ct.load(X, (k, block_y), (tileK, tileMN), order="F")

        acc = ct.mma(tile_x, tile_t, acc=acc)

    acc = acc * alpha
    acc = acc.astype(O.dtype)

    ct.store(O, (block_x, block_y), acc)
    if block_x != block_y:
        ct.store(O, (block_y, block_x), acc.transpose(0, 1))


@ct.kernel
def symm_matmul_bias(
    X: ct.Array,
    Y: ct.Array,
    O: ct.Array,
    alpha: float,
    beta: float,
    tileMN: ct.Constant,
    tileK: ct.Constant,
):
    # O = alpha * X @ X.T + beta * Y
    block_x, block_y = ct.bid(0), ct.bid(1)
    if block_y > block_x:
        return

    num_tile_k = ct.cdiv(X.shape[-1], tileK)
    acc = ct.load(Y, (block_x, block_y), (tileMN, tileMN)).astype(ct.float32)
    acc = acc * beta / alpha

    for k in range(num_tile_k):
        tile_x = ct.load(X, (block_x, k), (tileMN, tileK))
        tile_t = ct.load(X, (k, block_y), (tileK, tileMN), order="F")

        acc = ct.mma(tile_x, tile_t, acc=acc)

    acc = acc * alpha
    acc = acc.astype(O.dtype)
    ct.store(O, (block_x, block_y), acc)
    if block_x != block_y:
        ct.store(O, (block_y, block_x), acc.transpose(0, 1))


def muon_iteration(
    X: torch.Tensor,
    a: float,
    b: float,
    c: float,
    steps: int,
    tileMN: int = 64,
    tileK: int = 128,
):
    assert X.ndim == 2
    M, K = X.shape

    # Normalization
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for i in range(steps):
        # Compute bXX^T
        Y = torch.empty([M, M], dtype=X.dtype, device=X.device)
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(M, tileMN), ct.cdiv(M, tileMN), 1),
            symm_matmul,
            (X, Y, b, tileMN, tileK),
        )

        # Compute a * I + bXX.T + c * (XX.T)(XX.T).T
        # Z = c/b^2 * (bXX.T)(bXX.T).T + a * I
        Z = torch.eye(n=M, dtype=X.dtype, device=X.device)
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(M, tileMN), ct.cdiv(M, tileMN), 1),
            symm_matmul_bias,
            (Y, Z, Z, c / b / b, a, tileMN, tileK),
        )

        # Compute  (a * I + bXX.T + c * (XX.T)(XX.T).T)X
        # X = (Z + Y) @ X
        X = (Y + Z) @ X

    return X


if __name__ == "__main__":
    # 初始化参数
    M, K = 1024, 4096

    # 这里我们使用常用的牛顿-舒尔茨多项式系数 (例如 5 阶展开的变体)
    # 你也可以用最基础的: a=1.875, b=-1.25, c=0.375
    a, b, c = 3.4445, -4.7750, 2.0315
    steps = 5

    print(f"初始化矩阵形状: [{M}, {K}], 迭代步数: {steps}")

    # 构造相同的输入权重矩阵 (使用 bf16 以贴近真实大模型训练场景)
    torch.manual_seed(42)
    X_input = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # 1. 运行你的 cutile 优化算子
    # 克隆一份输入避免被 in-place 修改影响
    X_cutile = muon_iteration(X_input.clone(), a, b, c, steps, tileMN=64, tileK=128)

    print(X_cutile)
