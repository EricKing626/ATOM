#!/usr/bin/env bash
# =============================================================================
# Direction A — bring the 0.5.13.post1 kernel fixes into a 0.5.12 ATOM container
# without upgrading sglang python or touching ATOM.
#
# Rationale (verified from source): sglang's rope/hadamard PYTHON is identical
# between 0.5.12 and 0.5.13.post1 (rope still `from sgl_kernel import
# rotary_embedding`; hadamard still `from fast_hadamard_transform import ...`).
# The segfaults (barrier 1 rope, barrier 4 topk) are fixed only in the COMPILED
# sgl_kernel binary. So we rebuild ONLY sgl-kernel from the 0.5.13.post1 tree,
# keep sglang python at 0.5.12 and ATOM as-is (its validated combo), and add the
# two packaging fixes (barrier 2 hadamard, barrier 3 tilelang).
#
# Run INSIDE the running 0.5.12 atom_sglang container.
# =============================================================================
set -euo pipefail
SGLREF="${SGLREF:-v0.5.13.post1}"
ARCH="${AMDGPU_TARGET:-gfx942}"
VENV_PY="${VENV_PYTHON:-/opt/venv/bin/python}"
WORK="${WORK:-/tmp/sg513}"

echo "== [A0] current sgl_kernel / sglang =="
"$VENV_PY" - <<'PY'
import importlib.util as u
for m in ("sgl_kernel","sglang"):
    s=u.find_spec(m); print(m, "->", getattr(s,"origin",None))
import sglang; print("sglang", sglang.__version__)
PY

echo "== [A1] fetch sgl-kernel from ${SGLREF} =="
rm -rf "$WORK"
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang "$WORK"
git -C "$WORK" checkout "$SGLREF" -- sgl-kernel || {
  echo "sparse checkout failed; doing full checkout"; git -C "$WORK" checkout "$SGLREF"; }
test -d "$WORK/sgl-kernel" || { echo "ERROR: sgl-kernel dir missing"; exit 1; }

echo "== [A2] rebuild + install ONLY sgl-kernel for ${ARCH} =="
cd "$WORK/sgl-kernel"
AMDGPU_TARGET="$ARCH" "$VENV_PY" setup_rocm.py install

echo "== [A3] barrier 2 (hadamard) + barrier 3 (tilelang ABI) packaging =="
"$VENV_PY" -c "import fast_hadamard_transform" 2>/dev/null \
  || "$VENV_PY" -m pip install fast_hadamard_transform 2>/dev/null \
  || echo "NOTE: install failed; drop the pure-torch shim from setup_atom_sglang_v0513.sh"
"$VENV_PY" -m pip install --no-deps --force-reinstall "apache-tvm-ffi==0.1.10"
"$VENV_PY" -m pip install --no-deps "ml_dtypes==0.5.4" || true

echo "== [A4] smoke-test the two previously-crashing ops =="
"$VENV_PY" - <<'PY'
import torch, sgl_kernel
# rope (barrier 1)
q=torch.randn(4,64,dtype=torch.bfloat16,device='cuda'); k=q.clone()
pos=torch.arange(4,dtype=torch.int64,device='cuda')
cs=torch.randn(2048,64,dtype=torch.bfloat16,device='cuda')
torch.ops.sgl_kernel.rotary_embedding(pos,q,k,64,cs,False); torch.cuda.synchronize()
print("OK barrier1: sgl_kernel.rotary_embedding")
# topk op presence (barrier 4) — schema check
names=[n for n in dir(torch.ops.sgl_kernel) if "topk" in n.lower()]
print("topk ops available:", names or "<none - check op name/schema>")
PY

cat <<'EOF'

== [A5] next: launch GLM-5.2 NATIVE (no ATOM) then with ATOM ==
# native (proves the kernel swap fixed all 4 barriers):
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VENV_PY -m sglang.launch_server --model-path <GLM-5.2-FP8> --tp 8 \
    --trust-remote-code --max-running-requests 16
# with ATOM (0.5.12 python + ATOM unchanged + new kernel):
SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VENV_PY -m sglang.launch_server --model-path <GLM-5.2-FP8> --tp 8 --trust-remote-code

# WARNING: if launch/runtime errors with "undefined symbol" or an op-schema
# mismatch on a sgl_kernel op, the 0.5.13 kernel ABI diverged from 0.5.12
# python -> Direction A won't work cleanly; use Direction B (renovate ATOM to
# 0.5.13, see docker/ + RENOVATION_v0.5.13.md).
EOF
echo "Direction A done."
