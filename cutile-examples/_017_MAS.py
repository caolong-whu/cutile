import torch
import triton
import triton.language as tl
import Optional

import torch
import cuda.tile as ct


@ct.kernel
def ct_merge_attention_states(
    prefix_output: ct.Array,
    suffix_output: ct.Array,
    prefix_lse: ct.Array,
    suffix_lse: ct.Array,
    output: ct.Array,
    output_lse: ct.Array,
    dim: int,
    tile_size: ct.Constant,
    OUTPUT_LSE: ct.Constant,
):
    block_s = ct.bid(0)
    block_h = ct.bid(1)

    p_lse = ct.load(prefix_lse, (block_h, block_s), (1, 1))
    s_lse = ct.load(suffix_lse, (block_h, block_s), (1, 1))

    p_lse = ct.where(p_lse == float("inf"), float("-inf"), p_lse)
    s_lse = ct.where(s_lse == float("inf"), float("-inf"), s_lse)

    max_lse = ct.maximum(p_lse, s_lse)

    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse

    p_se = ct.exp(p_lse)
    s_se = ct.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = ct.log(out_se) + max_lse
        ct.store(output_lse, (block_h, block_s), out_lse.astype(output_lse.dtype))

    p_out = ct.load(
        prefix_output,
        (block_s, block_h, 0),
        (1, 1, tile_size),
        padding_mode=ct.PaddingMode.ZERO,
    )
    s_out = ct.load(
        suffix_output,
        (block_s, block_h, 0),
        (1, 1, tile_size),
        padding_mode=ct.PaddingMode.ZERO,
    )

    p_scale = p_se / out_se
    s_scale = s_se / out_se

    out = p_out * p_scale + s_out * s_scale

    ct.store(output, (block_s, block_h, 0), out.astype(output.dtype))


# ---------------------------------------------------------
# 2. PyTorch 参考实现 (用于对照精度)
# ---------------------------------------------------------
def reference_merge(p_out, s_out, p_lse, s_lse):
    # 克隆避免原地修改原数据
    p_lse = p_lse.clone()
    s_lse = s_lse.clone()

    p_lse[p_lse == float("inf")] = float("-inf")
    s_lse[s_lse == float("inf")] = float("-inf")

    max_lse = torch.maximum(p_lse, s_lse)
    p_lse_norm = p_lse - max_lse
    s_lse_norm = s_lse - max_lse

    p_exp = torch.exp(p_lse_norm)
    s_exp = torch.exp(s_lse_norm)
    out_se = p_exp + s_exp

    out_lse = torch.log(out_se) + max_lse

    # [NUM_HEADS, NUM_TOKENS] -> [NUM_TOKENS, NUM_HEADS, 1] 广播
    p_scale = (p_exp / out_se).transpose(0, 1).unsqueeze(-1)
    s_scale = (s_exp / out_se).transpose(0, 1).unsqueeze(-1)

    out = p_out * p_scale + s_out * s_scale
    return out, out_lse


# ---------------------------------------------------------
# 3. 驱动测试代码
# ---------------------------------------------------------
if __name__ == "__main__":
    # 配置参数：模拟真实推理环境
    num_tokens = 512  # seq_len
    num_heads = 32  # 注意力头数
    head_dim = 128  # 真实的 head 特征维度 (dim)
    tile_size = 256  # 硬件执行块大小 (必须 >= head_dim)

    # 1. 制造随机测试数据
    prefix_output = torch.randn(
        (num_tokens, num_heads, head_dim), dtype=torch.float32, device="cuda"
    )
    suffix_output = torch.randn(
        (num_tokens, num_heads, head_dim), dtype=torch.float32, device="cuda"
    )

    prefix_lse = torch.randn(
        (num_heads, num_tokens), dtype=torch.float32, device="cuda"
    )
    suffix_lse = torch.randn(
        (num_heads, num_tokens), dtype=torch.float32, device="cuda"
    )

    # 🚨 投毒测试：手动注入垃圾 inf 数据
    # 我们故意把第0个Head的第5个Token的前半段设为 inf，后半段设为正常值 10.0
    prefix_lse[0, 5] = float("inf")
    suffix_lse[0, 5] = 10.0

    # 2. 分配输出显存
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)

    # 3. 运行你的 cuTile 算子
    # grid 设置：由于 Kernel 里 block_s = ct.bid(0), block_h = ct.bid(1)
    # 所以第一个维度必须是 num_tokens，第二个维度必须是 num_heads
    grid = (num_tokens, num_heads)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_merge_attention_states,
        (
            prefix_output,
            suffix_output,
            prefix_lse,
            suffix_lse,
            output,
            output_lse,
            head_dim,
            tile_size,
            True,
        ),
    )

    # 4. 运行 PyTorch 对照组
    ref_output, ref_output_lse = reference_merge(
        prefix_output, suffix_output, prefix_lse, suffix_lse
    )

    # 5. 见证奇迹的时刻
    print("=" * 50)
    print("测试结果验证：")

    # 计算误差
    out_diff = (output - ref_output).abs().max().item()
    lse_diff = (output_lse - ref_output_lse).abs().max().item()

    print(f"Output 矩阵最大误差: {out_diff:.6e}")
    print(f"LSE 矩阵最大误差:   {lse_diff:.6e}")

    if out_diff < 1e-5 and lse_diff < 1e-5:
        print("\n✅ 测试完美通过！你的底层算子逻辑和数学精度完全正确！")
    else:
        print("\n❌ 存在数值误差，请检查代码！")

    # 看看毒药被解除了没
    print("\n[边缘情况验证 (inf 解毒测试)]:")
    print(f"PyTorch Ref  (Head 0, Token 5) LSE: {ref_output_lse[0, 5].item():.4f}")
    print(f"Your Kernel  (Head 0, Token 5) LSE: {output_lse[0, 5].item():.4f}")
