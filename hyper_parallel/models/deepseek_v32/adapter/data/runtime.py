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
"""Forward global packed boundaries to DeepSeek-V3.2 DSA kernels."""

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext


class DeepseekV32Runtime(RuntimeInputAdapter):
    """Preserve document isolation after the generic batch runtime shards tokens.

    ``actual_seq_len`` contains global cumulative document ends, without the
    leading zero. It is deliberately not sliced with CP-local token IDs. Each
    attention layer derives its local Q/K prefixes from these same host ends.
    Position IDs remain owned by ``TextParallelBatch``.
    """

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the extra field consumed by the DSA replacement."""
        return ("actual_seq_len",)

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Validate and copy cumulative metadata to the host once per batch."""
        if len(context.local_input_shape) != 2:
            raise ValueError("DeepSeek-V3.2 runtime requires local input shape [batch, sequence]")
        batch_size, local_length = context.local_input_shape
        global_length = local_length * context.parallel_sizes.get("cp", 1)
        if batch_size < 1 or global_length < 1:
            raise ValueError("DeepSeek-V3.2 runtime requires nonempty input rows")
        boundaries = batch.get("cu_seq_lens")
        if isinstance(boundaries, torch.Tensor):
            if boundaries.ndim != 1 or boundaries.dtype not in (torch.int32, torch.int64):
                raise ValueError("cu_seq_lens must be a one-dimensional integer tensor")
            boundaries = boundaries.tolist()
        if not isinstance(boundaries, (tuple, list)) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
            raise ValueError("cu_seq_lens must contain integer cumulative boundaries")
        if (len(boundaries) < 2 or boundaries[0] != 0
                or boundaries[-1] != batch_size * global_length
                or any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:]))):
            raise ValueError("cu_seq_lens must start at zero, strictly increase and cover the global batch")
        if any(start // global_length != (end - 1) // global_length
               for start, end in zip(boundaries[:-1], boundaries[1:])):
            raise ValueError("DSA packed documents cannot cross physical batch rows")
        return {"actual_seq_len": tuple(boundaries[1:])}


__all__ = ["DeepseekV32Runtime"]
