import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinkhorn_knopp(x: torch.Tensor, num_iters: int=1, eps: float=1e-20) -> torch.Tensor:
    x_exp = torch.exp(x)
    
    for i in range(num_iters):
        x_exp = x_exp / (x_exp.sum(dim=-1, keepdim=True) + eps)    
        x_exp = x_exp / (x_exp.sum(dim=-2, keepdim=True) + eps)
        
    return x_exp
        
    

class mHC(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.dim = dim
        self.n = n
        self.nc = n * dim
        self.n2 = n * n
        
        self.phi = nn.Linear(self.nc, self.n2 + 2 * self.n, bias=False)
        
        self.alpha = nn.Parameter(torch.ones((3, )) * 0.01)
        self.b = nn.Parameter(torch.zeros((self.n2 + 2 * self.n)))
    
    def width_connection(self, hidden_states):
        # hidden_states: [B, L, n, dim]
        B, L, L, D = hidden_states.shape
        
        hidden_states_flat = hidden_states.flatten(2) # [B, L ,n*dim]
        
        """
        RMSNorm(x): gamma * x / rms(x)
        H = phi * RMSNorm(x) = phi * gamma * x / rms(x)
        gamma参数可以吸收进phi中，只需要计算rms(x)
        """
        r = hidden_states_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        
        H = self.phi(hidden_states) # [B, L, n2 + 2 * n]
        
        H_pre = 1 / r * self.alpha[0] * H[:, :, :self.n] + self.b[:self.n] # [B, L, n]
        H_post = 1 / r * self.alpha[1] * H[:, :, self.n:2*self.n] + self.b[self.n:2*self.n] # [B, L, n]
        H_res = 1 / r * self.alpha[2] * H[:, :, 2*self.n:] + self.b[2*self.n:] # [B, L, n*n]
        
        H_pre = F.sigmoid(H_pre)
        H_post = 2 * F.sigmoid(H_post)
        
        H_res = H_res.reshape(B, L, self.n, self.n) # [B, L, n, n]
        H_res = sinkhorn_knopp(H_res)
        
        h_pre = H_pre.unsqueeze(2) @ hidden_states # [B, L, 1, dim]
        h_res = H_res @ hidden_states # [B, L, n, dim]
        
        return h_pre, h_res, H_post
    
    def depth_connection(self, h_res, hidden_states, H_post):
        # h_res: [B, L, n, dim] 
        # hidden_states: [B, L, dim] 
        # H_post: [B, L, n]
        h_post = H_post.unsqueeze(-1) @ hidden_states.unsqueeze(-2) # [B, L, n, dim]
        output = h_post + h_res
        
        return output

    
import torch
import torch.nn as nn
import math
import cuda.tile as ct
torch.cuda.empty_cache()

print(torch.cuda.memory_summary())
INV_LOG_2 = 1.0 / math.log(2)

def sinkhorn_knopp(x: torch.Tensor, num_iters: int=1, eps: float=1e-20) -> torch.Tensor:
    x_exp = torch.exp(x)
    
    for i in range(num_iters):
        x_exp = x_exp / (x_exp.sum(dim=-1, keepdim=True) + eps)    
        x_exp = x_exp / (x_exp.sum(dim=-2, keepdim=True) + eps)
        
    return x_exp

@ct.function
def sigmoid_exp2(X: ct.Array):
    return 1 / (1 + ct.exp2(-1 * X))

@ct.function
def sinkhorn_exp2(tile: ct.Tile, iter: int=20):
    tile = ct.exp2(tile)
    for i in range(iter):
        tile = tile / ct.sum(tile, axis=-1, keepdims=True)
        tile = tile / ct.sum(tile, axis=-2, keepdims=True)
    return tile

@ct.kernel
def Fused_Compute_H_Matrix_Kernel(
    X: ct.Array, # [M ,K]
    phi: ct.Array, # [K, 32]
    H: ct.Array, # [2, M, 32]
    Stream: ct.Constant,
    tileM: ct.Constant,
    tileK: ct.Constant,
    Chunk_Size: ct.Constant
):
    _, K = X.shape
    N: ct.Constant = 32
    
    n_routed: ct.Constant = 32
    
    block_m, block_k = ct.bid(0), ct.bid(1)
    
    num_tileK_per_chunk = ct.cdiv(Chunk_Size, tileK)
    
    acc_H = ct.full((tileM, N), 0.0, dtype=ct.float32)
    square_sum_x = ct.full((tileM, 1), 0.0, dtype=ct.float32)
    
    for iterK in range(num_tileK_per_chunk):
        
        x_tile = ct.load(X, (block_m, block_k * num_tileK_per_chunk + iterK), (tileM, tileK), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
        
        phi_tile = ct.load(phi, (block_k * num_tileK_per_chunk + iterK, 0), (tileK, n_routed), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
        
        acc_H += x_tile @ phi_tile # [tileM, tileK] @ [tileK, n_routed] -> [tileM, n_routed]
        
        square_sum_x += ct.sum(x_tile * x_tile, axis=-1, keepdims=True)
        
    ct.store(H, (block_k, block_m, 0), acc_H.reshape((1, tileM, N)).astype(H.dtype))
    ct.store(H, (block_k, block_m, n_routed), square_sum_x.reshape((1, tileM, 1)).astype(H.dtype))

@ct.kernel
def Split_H_Kernel(
    H: ct.Array, # [2, M, 32]
    AlphaBeta: ct.Array, # [32]
    H_pre: ct.Array, # [M, K // 4]
    H_res: ct.Array, # [M, 4, 4]
    H_post: ct.Array, # [M, 4]
    NCHUNKS: ct.Constant,
    D: ct.Constant,
    tileM: ct.Constant,
):
    tileN: ct.Constant = 32
    Stream: ct.Constant = 4
    block_m, block_k = ct.bid(0), ct.bid(1)
    
    raw = ct.load(AlphaBeta, (0, ), (tileN, ), padding_mode=ct.PaddingMode.ZERO)
    
    offset = Stream * Stream + 2 * Stream
    alpha_pre = ct.extract(raw, (offset, ), (1, )).reshape((1, 1))
    alpha_res = ct.extract(raw, (offset + 1, ), (1, )).reshape((1, 1, 1))
    alpha_post = ct.extract(raw, (offset + 2, ), (1, )).reshape((1, 1))
    beta_pre = ct.extract(raw, (0, ), (Stream, )).reshape((1, Stream))
    beta_res = ct.extract(raw, (Stream, ), (Stream * Stream, )).reshape((1, Stream, Stream))
    beta_post = ct.extract(raw, (Stream * Stream + Stream,), (Stream, )).reshape((1, Stream))
    
    # n_chunks = H.shape[0]
    acc_H = ct.full((1, tileM, tileN), 0.0, dtype=ct.float32)
    for i in range(NCHUNKS):
        H_tile = ct.load(H, (i, block_m, 0), (1, tileM, tileN))
        acc_H += H_tile
    
    acc_H = INV_LOG_2 * acc_H.reshape((tileM, tileN))
    
    H_pre_tile =  ct.extract(acc_H, (0, 0), (tileM, Stream))
    H_res_tile = ct.extract(acc_H, (0, Stream), (tileM, Stream * Stream)).reshape((tileM, Stream, Stream))
    H_post_tile = ct.extract(acc_H, (0, Stream * Stream + Stream), (tileM, Stream))
    
    rms_norm_tile = ct.rsqrt(ct.extract(acc_H, (0, 32), (tileM, 1)) / D + 1e-9)
    
    H_pre_tile =  sigmoid_exp2(rms_norm_tile * alpha_pre * H_pre_tile + beta_pre)
    H_res_tile = ct.exp2(rms_norm_tile.reshape((tileM, 1, 1)) * alpha_res * H_res_tile + beta_res)
    for i in range(1):
        H_res_tile = H_res_tile / (ct.sum(H_res_tile, axis=-1, keepdims=True) + 1e-20)    
        H_res_tile = H_res_tile / (ct.sum(H_res_tile, axis=-2, keepdims=True) + 1e-20)
    H_post_tile = 2 * sigmoid_exp2(rms_norm_tile * alpha_post * H_post_tile + beta_post)
    
    ct.store(H_pre, (block_m, 0), H_pre_tile.astype(H_pre.dtype), allow_tma=False)
    ct.store(H_res, (block_m, 0, 0), H_res_tile.astype(H_res.dtype), allow_tma=False)
    ct.store(H_post, (block_m, 0), H_post_tile.astype(H_post.dtype), allow_tma=False)
     
@ct.kernel
def Apply_Residual_Kernel(
    X: ct.Array,
    H_res: ct.Array,
    X_pre: ct.Array,
    H_post: ct.Array,
    O: ct.Array,
    Stream: ct.Constant,
    tileM: ct.Constant,
    tileK: ct.Constant
):
    # X: [M, 4, K // 4]
    # H_res: [M, 4, 4]
    # X_pre: [M, K // 4]
    # H_post: [M, 4]
    # O: [M, 4, K // 4] 
    # O = H_res[block_m, 4, 4] @ X[block_m, 4, K // 4] + H_post[block_m, 4] * X_pre[block_m, K // 4]
    block_m, block_k = ct.bid(0), ct.bid(1)
    tile_x = ct.load(X, (block_m, 0, block_k), (1, 4, tileK))
    tile_h_res = ct.load(H_res, (block_m, 0, 0), (1, 4, 4))
    
    tile_x_res = tile_h_res @ tile_x # [1, 4, tileK]
    
    tile_h_post = ct.load(H_post, (block_m, 0), (1, 4)).reshape((1, 4, 1))
    tile_x_pre = ct.load(X_pre, (block_m, block_k), (1, tileK)).reshape((1, 1, tileK))
    tile_x_post = tile_h_post * tile_x_pre
    
    
    out = tile_x_post + tile_x_res
    ct.store(O, (block_m, 0, block_k), out.astype(O.dtype))
    
@ct.kernel
def ApplyPreTransform_Kernel(
    X: ct.Array, # [M, 4, K // 4]
    H_pre: ct.Array, # [M, 4]
    O: ct.Array, # [M, K // 4]
    tile_size: ct.Constant
):
    block_m, block_y = ct.bid(0), ct.bid(1)
    
    tile_x = ct.load(X, (block_m, 0, block_y), (1, 4, tile_size), padding_mode=ct.PaddingMode.ZERO, allow_tma=False)
    tile_h = ct.load(H_pre, (block_m, 0), (1, 4), allow_tma=False)
    tile_h = tile_h.reshape((1, 4, 1))
    tile_y = ct.sum(tile_h * tile_x, axis=1).reshape((1, tile_size))
    ct.store(O, (block_m, block_y), tile_y.astype(O.dtype), allow_tma=False)

  
def Compute_H_RmsNorm(x, phi, n_stream: int=4, chunk_size: int=512, tileM: int=128, tileK: int=128):
    M, K = x.shape
    K, N = phi.shape
    
    assert K % chunk_size == 0
    
    H = torch.empty([ct.cdiv(K, chunk_size), M, N], dtype=torch.float32, device="cuda")
    
    grid = (ct.cdiv(M, tileM), ct.cdiv(K, chunk_size))
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        Fused_Compute_H_Matrix_Kernel,
        (x, phi, H, n_stream, tileM, tileK, chunk_size)
    )
    
    return H

def Split_H(H, X, alpha_beta, tileM: int=128, stream: int=4, tileK: int=1024):
    
    M, K = X.shape
    n_chunks, _, N = H.shape
    assert N == alpha_beta.shape[0]
    
    H_pre = torch.empty([M, stream], dtype=torch.float32, device=H.device)
    H_res = torch.empty([M, stream, stream], dtype=torch.float32, device=H.device)
    H_post = torch.empty([M, stream], dtype=torch.float32, device=H.device)
    
    grid = (ct.cdiv(M, tileM), )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        Split_H_Kernel,
        (H, alpha_beta, H_pre, H_res, H_post, n_chunks, K // stream, tileM)
    )
    
    return H_pre, H_res, H_post

def Apply_Residual(X, H_res, X_pre, H_post, tileM: int=128, tileK: int=256):
    # X: [M, K]
    # H_res: [M, 4, 4]
    # X_pre: [M, K // 4]
    # H_post: [M, 4]
    
    # O = H_res[block_m, 4, 4] @ X[block_m, 4, K // 4] + H_post[block_m, 4] * X_pre[block_m, K // 4]
    M, K = X.shape
    _, D = X_pre.shape
    _, Stream = H_post.shape
    
    O =  torch.empty([M , Stream, K // 4], dtype=X.dtype, device=X.device)

    grid = (M, ct.cdiv(D, tileK))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        Apply_Residual_Kernel,
        (X.view(M, Stream, K // Stream), H_res, X_pre, H_post, O, Stream, tileM, tileK)
    )
    return O.view(M, -1)
    
def Apply_Pre_Transformer(X: torch.Tensor, H_pre: torch.Tensor, tile_size: int=256):
    M, K = X.shape
    assert K % 4 == 0
    Y = torch.empty(size=[M, K // 4], device=X.device, dtype=X.dtype)
    
    # X: [M, K]
    # H_pre: [M, 4]
    # O: [M, K // 4] = H_pre[block_m, 4]  X[block_m, 4, block_y] 
    ct.launch(
        torch.cuda.current_stream(), 
        (M, ct.cdiv(K // 4, tile_size)),
        ApplyPreTransform_Kernel,
        (X.view(M, 4, K // 4), H_pre, Y, tile_size)
    )
    return Y

class mHC(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        
        self.dim = dim
        self.n = n
        self.nc = n * dim
        self.n2 = n * n
        
        alpha = torch.ones((8, )) * 0.01
        beta = torch.zeros((self.n2 + 2 * self.n))
        self.alpha_beta = nn.Parameter(torch.cat([beta, alpha], dim=0)) # [32]
        self.phi = nn.Parameter(torch.randn((self.dim * self.n, 32), dtype=torch.float32)) # [n*dim, 32]
        
    def forward(self, x: torch.Tensor, chunk_size: int=512):
        
        # H_chunked: [2, M, 32]
        H = Compute_H_RmsNorm(x, self.phi, self.n, chunk_size=chunk_size)
        
        print("Compute_H_RmsNorm Pass!")
        
        
        H_pre, H_res, H_post = Split_H(H, x, self.alpha_beta)
        
        print("Split_H Pass!")
        
        X_pre = Apply_Pre_Transformer(x, H_pre)
        # X_pre = H_pre.unsqueeze(1) @ x.view(x.shape[0], 4, -1)
        # X_pre = X_pre.squeeze(1)
        
        print("Apply_Pre_Transformer Pass!")
        
        out = Apply_Residual(x, H_res, X_pre, H_post)
        print("Apply_Residual Pass!")
        
        return out
    
    def reference_logic(self, X: torch.Tensor):

        M, K = X.shape
        D = K // 4
        
        # 1. 计算 RMSNorm 的缩放因子 (注意保持维度以便广播)
        # rmsnorm 形状应为 [M, 1]
        rmsnorm = (X.norm(dim=-1, keepdim=True) / math.sqrt(K)) + 1e-9
        inv_rmsnorm = 1.0 / rmsnorm

        # 2. 线性投影
        H = torch.matmul(X, self.phi) 
        
        # 3. 提取超参和权重
        H_res, H_pre, H_pos = H[:, 0:16], H[:, 16:20], H[:, 20:24]
        a_res, a_pre, a_pos = self.alpha_beta[24:25], self.alpha_beta[25:26], self.alpha_beta[26:27]
        b_res, b_pre, b_pos = self.alpha_beta[0:16], self.alpha_beta[16:20], self.alpha_beta[20:24]
        
        # 4. 【核心修改】：先缩放，再激活
        # 对于 H_pre 和 H_pos，缩放因子 inv_rmsnorm 必须在 torch.sigmoid 内部
        H_pre = torch.sigmoid(inv_rmsnorm * a_pre * H_pre + b_pre)
        H_pos = 2.0 * torch.sigmoid(inv_rmsnorm * a_pos * H_pos + b_pos)
        
        # 对于 H_res (线性混合)，缩放位置相对不敏感，但为了对齐内核逻辑，通常也先缩放
        H_res = sinkhorn_knopp(inv_rmsnorm.unsqueeze(-1) * a_res * H_res.view(M, 4, 4)  + b_res.reshape(1, 4, 4))

        # 5. 维度变换
        H_pre = H_pre.reshape(M, 1, 4)   # [M, 1, 4]
        # H_res = H_res.reshape(M, 4, 4)   # [M, 4, 4] 
        H_pos = H_pos.reshape(M, 4, 1)   # [M, 4, 1]
        
        X_view = X.reshape(M, 4, D)      # [M, 4, D]
        
        # 6. 特征混合
        # X_res: [M, 4, 4] @ [M, 4, D] -> [M, 4, D]
        X_res_out = torch.bmm(H_res, X_view) 
        
        # X_pre: [M, 1, 4] @ [M, 4, D] -> [M, 1, D]
        X_pre_out = torch.bmm(H_pre, X_view) 
        
        # X_pos: [M, 4, 1] @ [M, 1, D] -> [M, 4, D]
        X_pos_out = torch.bmm(H_pos, X_pre_out) 
        
        # 7. 合并输出
        out = X_pos_out + X_res_out
        return out.view(M, K)
    
MHC = mHC(dim=256, n=4).to(device="cuda")
X = torch.randn(size=[256, 256*4], device="cuda", dtype=torch.float32)
pred = MHC(X)
real = MHC.reference_logic(X)
print(pred - real)