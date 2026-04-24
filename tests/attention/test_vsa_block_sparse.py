"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.utils import is_sm100a_supported

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="VSA Blackwell backend requires sm100a (Blackwell GPU)",
)

# VSA Blackwell kernel has fixed constraints:
R = C = 64
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_random_bsr(MB: int, NB: int, density: float, device: torch.device):
    """Return (indptr, indices) for a random BSR pattern; every row has >= 1 block."""
    rows = []
    for _ in range(MB):
        k = max(1, int(round(density * NB)))
        k = min(k, NB)
        col_indices = torch.randperm(NB, device="cpu")[:k].sort().values
        rows.append(col_indices)

    indptr = torch.zeros(MB + 1, dtype=torch.int32)
    indices_list = []
    for i, row in enumerate(rows):
        indptr[i + 1] = indptr[i] + len(row)
        indices_list.append(row)

    indices = torch.cat(indices_list).to(torch.int32)
    return indptr.to(device), indices.to(device)


def _bsr_to_dense_mask(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    MB: int,
    NB: int,
    R: int,
    C: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand BSR sparsity pattern into a token-level boolean mask [M, N]."""
    mask = torch.zeros(MB * R, NB * C, dtype=torch.bool, device=device)
    indptr_cpu = indptr.cpu()
    indices_cpu = indices.cpu()
    for i in range(MB):
        s, e = int(indptr_cpu[i]), int(indptr_cpu[i + 1])
        for j_blk in indices_cpu[s:e].tolist():
            mask[i * R : i * R + R, j_blk * C : j_blk * C + C] = True
    return mask


def _pytorch_ref(
    q: torch.Tensor,     # [M, H, D]
    k: torch.Tensor,     # [N, H, D]
    v: torch.Tensor,     # [N, H, D]
    indptr: torch.Tensor,
    indices: torch.Tensor,
    R: int,
    C: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """Dense PyTorch reference for block-sparse attention."""
    M, H, D = q.shape
    N = k.shape[0]
    MB, NB = M // R, N // C
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    mask = _bsr_to_dense_mask(indptr, indices, MB, NB, R, C, q.device)

    qf = q.float().permute(1, 0, 2)           # [H, M, D]
    kf = k.float().permute(1, 0, 2)           # [H, N, D]
    vf = v.float().permute(1, 0, 2)           # [H, N, D]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale  # [H, M, N]
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)             # [H, M, D]
    return out.permute(1, 0, 2).to(q.dtype)  # [M, H, D]


def _make_wrapper(device):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("num_blocks", [4, 8, 16])
@pytest.mark.parametrize("density", [0.25, 0.75])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_vsa_accuracy(num_heads, num_blocks, density, dtype):
    """VSA output must match PyTorch dense reference."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    M = N = num_blocks * R
    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)

    indptr, indices = _build_random_bsr(num_blocks, num_blocks, density, device)

    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


def test_vsa_preallocated_output():
    """run(out=...) must write into the provided tensor."""
    device = torch.device("cuda")
    torch.manual_seed(1)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )

    o = wrapper.run(q, k, v)
    o_buf = torch.empty_like(o)
    wrapper.run(q, k, v, out=o_buf)
    torch.testing.assert_close(o, o_buf)


def test_vsa_return_lse():
    """return_lse=True must yield correctly shaped, finite LSE."""
    device = torch.device("cuda")
    torch.manual_seed(2)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )

    o, lse = wrapper.run(q, k, v, return_lse=True)

    assert o.shape == (M, num_heads, HEAD_DIM)
    assert lse.shape == (M, num_heads)
    assert lse.dtype == torch.float32
    assert not torch.isnan(lse).any()
    assert not torch.isinf(lse).any()


def test_vsa_preallocated_lse():
    """run(lse=...) must write LSE into the provided tensor."""
    device = torch.device("cuda")
    torch.manual_seed(3)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )

    _, lse = wrapper.run(q, k, v, return_lse=True)
    lse_buf = torch.empty(M, num_heads, dtype=torch.float32, device=device)
    _, lse2 = wrapper.run(q, k, v, lse=lse_buf, return_lse=True)

    assert lse2 is lse_buf
    torch.testing.assert_close(lse, lse_buf)


@pytest.mark.parametrize("sm_scale", [0.05, 0.1, 0.5])
def test_vsa_sm_scale(sm_scale):
    """User-supplied sm_scale must propagate correctly."""
    device = torch.device("cuda")
    torch.manual_seed(4)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    o_ref = _pytorch_ref(q, k, v, indptr, indices, R, C, sm_scale=sm_scale)

    wrapper = _make_wrapper(device)
    wrapper.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
        sm_scale=sm_scale,
    )
    o = wrapper.run(q, k, v)

    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-2)


def test_vsa_vs_flashinfer_default_backend():
    """Cross-validate VSA backend against FlashInfer's default backend."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    num_heads, num_blocks = 4, 8
    M = N = num_blocks * R
    dtype = torch.bfloat16

    q = torch.randn(M, num_heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(N, num_heads, HEAD_DIM, dtype=dtype, device=device)
    indptr, indices = _build_random_bsr(num_blocks, num_blocks, 0.5, device)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref_w = BlockSparseAttentionWrapper(ws, backend="auto")
    ref_w.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o_ref = ref_w.run(q, k, v)

    vsa_w = BlockSparseAttentionWrapper(ws, backend="vsa_blackwell")
    vsa_w.plan(
        indptr, indices, M, N, R, C,
        num_heads, num_heads, HEAD_DIM,
        q_data_type=dtype,
    )
    o_vsa = vsa_w.run(q, k, v)

    torch.testing.assert_close(o_ref.float(), o_vsa.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# plan() constraint-violation tests
# ---------------------------------------------------------------------------


def _plan_indptr_indices(device, num_blocks=2):
    indptr = torch.arange(num_blocks + 1, dtype=torch.int32, device=device)
    indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return indptr, indices


@pytest.mark.parametrize(
    "bad_R, bad_C",
    [(32, 32), (64, 32), (32, 64), (128, 128)],
)
def test_plan_rejects_bad_block_size(bad_R, bad_C):
    device = torch.device("cuda")
    indptr, indices = _plan_indptr_indices(device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="R == C == 64"):
        wrapper.plan(indptr, indices, bad_R * 2, bad_C * 2, bad_R, bad_C, 4, 4, HEAD_DIM)


def test_plan_rejects_bad_head_dim():
    device = torch.device("cuda")
    indptr, indices = _plan_indptr_indices(device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="head_dim == 128"):
        wrapper.plan(indptr, indices, 2 * R, 2 * C, R, C, 4, 4, 64)


def test_plan_rejects_gqa():
    device = torch.device("cuda")
    indptr, indices = _plan_indptr_indices(device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="GQA"):
        wrapper.plan(indptr, indices, 2 * R, 2 * C, R, C, 8, 4, HEAD_DIM)


def test_plan_rejects_mask():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    nnz = int(indices.shape[0])
    mask = torch.ones(nnz, R, C, dtype=torch.bool, device=device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="mask"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            mask=mask,
        )


def test_plan_rejects_packed_mask():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    packed_mask = torch.zeros(1, dtype=torch.uint8, device=device)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="mask"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            packed_mask=packed_mask,
        )


def test_plan_rejects_causal():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="causal"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            causal=True,
        )


def test_plan_rejects_pos_encoding_mode():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="pos_encoding_mode"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            pos_encoding_mode="ROPE_LLAMA",
        )


def test_plan_rejects_logits_soft_cap():
    device = torch.device("cuda")
    num_blocks = 2
    indptr, indices = _plan_indptr_indices(device, num_blocks)
    wrapper = _make_wrapper(device)
    with pytest.raises(ValueError, match="logits_soft_cap"):
        wrapper.plan(
            indptr, indices, num_blocks * R, num_blocks * C, R, C,
            4, 4, HEAD_DIM,
            logits_soft_cap=50.0,
        )
