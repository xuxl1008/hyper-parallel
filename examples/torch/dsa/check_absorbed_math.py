# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FP64 fixed-TopK absorbed/expanded MLA reference; not a training implementation."""

import json

import torch


def main() -> None:
    """Compare both formulas, including gradients of latent and up-projection weights."""
    torch.set_num_threads(1)
    torch.manual_seed(362)
    sequence, heads, latent, plain, rotary, value, topk = 32, 4, 16, 8, 4, 8, 7
    leaves = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in (
        (sequence, heads, plain), (sequence, latent), (sequence, heads, rotary), (sequence, rotary),
        (heads, plain, latent), (heads, value, latent),
    )]
    query, compressed, query_rope, key_rope, key_weight, value_weight = leaves
    indices = torch.full((sequence, topk), -1, dtype=torch.int64)
    for row in range(sequence):
        count = min(topk, row + 1)
        indices[row, :count] = torch.randperm(row + 1)[:count]
    valid, selected = indices >= 0, indices.clamp_min(0)
    upstream = torch.randn(sequence, heads, value, dtype=torch.float64)
    scale = (plain + rotary)**-0.5
    key = torch.einsum("sc,hdc->shd", compressed, key_weight)
    values = torch.einsum("sc,hvc->shv", compressed, value_weight)
    rotary_scores = torch.einsum("qhr,qjr->qhj", query_rope, key_rope[selected])
    expanded_scores = (torch.einsum("qhd,qjhd->qhj", query, key[selected]) + rotary_scores) * scale
    expanded_prob = expanded_scores.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
    expanded_output = torch.einsum("qhj,qjhv->qhv", expanded_prob, values[selected])
    absorbed_query = torch.einsum("qhd,hdc->qhc", query, key_weight)
    absorbed_scores = (torch.einsum("qhc,qjc->qhj", absorbed_query, compressed[selected]) + rotary_scores) * scale
    absorbed_prob = absorbed_scores.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
    absorbed_output = torch.einsum("qhc,hvc->qhv", torch.einsum("qhj,qjc->qhc", absorbed_prob,
                                                              compressed[selected]), value_weight)
    expanded_grads = torch.autograd.grad((expanded_output * upstream).sum(), leaves, retain_graph=True)
    absorbed_grads = torch.autograd.grad((absorbed_output * upstream).sum(), leaves)
    names = ("output", "query", "latent", "query_rope", "key_rope", "key_up_weight", "value_up_weight")
    errors = {}
    for name, actual, reference in zip(names, (absorbed_output, *absorbed_grads), (expanded_output, *expanded_grads)):
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
        errors[name] = float((actual.detach() - reference.detach()).abs().max())
    print(json.dumps({"completed": True, "max_abs": errors}, indent=2))


if __name__ == "__main__":
    main()
