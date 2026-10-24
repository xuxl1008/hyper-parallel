# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental sigmoid-gated RMSNorm with FP32 arithmetic and compact saves.

Unlike SiLU-gated upstream norms, this preserves Kimi's sigmoid gate. This
diagnostic supports only contiguous per-head D128 normalization on NPU.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _forward(X, Z, W, Y, R, N: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    mask = rows[:, None] < N
    x = tl.load(X + rows[:, None] * 128 + cols[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + rows[:, None] * 128 + cols[None, :], mask, 0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / 128 + EPS)
    y = ((x * r[:, None]) * w[None, :]) * tl.sigmoid(z)
    tl.store(Y + rows[:, None] * 128 + cols[None, :], y, mask)
    tl.store(R + rows, r, rows < N)


@triton.jit
def _backward(X, Z, W, R, DY, DX, DZ, DW, N: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    mask = rows[:, None] < N
    offset = rows[:, None] * 128 + cols[None, :]
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    z = tl.load(Z + offset, mask, 0).to(tl.float32)
    dy = tl.load(DY + offset, mask, 0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    r = tl.load(R + rows, rows < N, 0)
    xn = x * r[:, None]
    sig = tl.sigmoid(z)
    dxn = (dy * sig) * w[None, :]
    dot = tl.sum(dxn * xn, 1) / 128
    dx = (dxn - xn * dot[:, None]) * r[:, None]
    dz = (dy * (xn * w[None, :])) * sig * (1 - sig)
    dw = tl.sum((dy * sig) * xn, 0)
    tl.store(DX + offset, dx, mask)
    tl.store(DZ + offset, dz, mask)
    tl.store(DW + pid * 128 + cols, dw)


class GatedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps):
        if x.device.type != "npu" or x.shape != gate.shape or x.shape[-1] != 128:
            raise ValueError("Experimental gated norm requires identical NPU D128 inputs")
        if weight.shape != (128,) or x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Experimental gated norm requires D128 weight and BF16/FP32 inputs")
        if x.numel() == 0 or any(t.device != x.device or t.dtype not in (torch.bfloat16, torch.float32)
                                for t in (gate, weight)):
            raise ValueError("Experimental gated norm requires nonempty inputs on one NPU")
        x, gate = x.contiguous(), gate.contiguous()
        weight = weight.contiguous()
        rows = x.numel() // 128
        out = torch.empty_like(x)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        _forward[(triton.cdiv(rows, 32),)](x, gate, weight, out, rstd, rows, eps, 32,
                                         num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, gate, weight, rstd)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, gate, weight, rstd = ctx.saved_tensors
        grad = grad.contiguous()
        rows = x.numel() // 128
        blocks = triton.cdiv(rows, 32)
        dx, dz = torch.empty_like(x), torch.empty_like(gate)
        partial = torch.empty((blocks, 128), dtype=torch.float32, device=x.device)
        _backward[(blocks,)](x, gate, weight, rstd, grad, dx, dz, partial, rows, 32,
                            num_warps=4, enable_fp_fusion=False)
        return dx, dz, partial.sum(0).to(weight.dtype), None


class ExperimentalGatedNorm(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, hidden, gate):
        return GatedNorm.apply(hidden, gate, self.weight, self.variance_epsilon)
