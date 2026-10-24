#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$ROOT/../../.." && pwd)
: "${CANN_HOME:?Set CANN_HOME to the qualified CANN installation}"
: "${FLA_NPU_INSTALL:?Set FLA_NPU_INSTALL to the separately installed patched wheel}"
PYTHON=${PYTHON:-python3}
if [[ ! -f "$FLA_NPU_INSTALL/fla_npu/__init__.py" ]]; then
    printf 'Patched fla_npu installation not found: %s\n' "$FLA_NPU_INSTALL" >&2
    exit 1
fi
set +u
source "$CANN_HOME/set_env.sh"
set -u
VENDOR="$FLA_NPU_INSTALL/fla_npu/opp/vendors/fla_npu_transformer"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR:$VENDOR/op_api/lib"
export LD_LIBRARY_PATH="$CANN_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$FLA_NPU_INSTALL${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 USE_TF=0 OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=64 HCCL_EXEC_TIMEOUT=300
"$PYTHON" -c 'import importlib.util, pathlib, sys
spec = importlib.util.find_spec("hyper_parallel")
expected = pathlib.Path(sys.argv[1]) / "hyper_parallel/__init__.py"
if spec is None or pathlib.Path(spec.origin).resolve() != expected.resolve():
    raise SystemExit("Install this reference checkout with: python -m pip install -e . --no-deps")' "$REPO"
exec "$PYTHON" "$@"
