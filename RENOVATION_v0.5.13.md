# ATOM renovation for sglang v0.5.13.post1 (GLM-5.2-FP8 DSA on gfx942)

Goal: make ATOM's sglang plugin load & run on **sglang v0.5.13.post1** (whose
ROCm `sgl_kernel` fixes the rope/topk segfaults), instead of being pinned to
v0.5.12 (whose kernels crash on gfx942).

## Why 0.5.13.post1 (evidence)
- Isolated op test on the sgl-dev v0.5.13.post1 image: `torch.ops.sgl_kernel.
  rotary_embedding(...)` returns "rope OK" on gfx942. 0.5.12's build segfaults.
- rope/hadamard PYTHON is byte-identical between 0.5.12 and 0.5.13.post1; the fix
  is in the compiled `sgl_kernel` binary + shipping `fast_hadamard_transform`.
- Known-good dep set for 0.5.13.post1 (from sgl-dev `pip show`):
  `huggingface_hub==1.20.1`, `transformers==5.8.1`.

## Import audit: ATOM plugin vs v0.5.13.post1 (48 unique sglang imports)
Only **two** hard breaks — both pure renames (NSA -> DSA subsystem rename):

| symbol (0.5.12) | 0.5.13.post1 | file |
|---|---|---|
| `is_deepseek_nsa` (model_config) | renamed `is_deepseek_dsa` | `srt/configs/model_config.py:102` |
| `nsa_use_prefill_cp` (nsa/utils) | renamed `mla_use_prefill_cp` | `srt/layers/utils/cp_utils.py:93` |

Everything else still resolves (e.g. `sglang.srt.utils` became a package but its
`__init__.py` does `from ...common import *`, so `LazyValue/bind_or_assign/
is_gfx95_supported/get_bool_env_var` still import). `get_is_capture_mode` is still
in `cuda_graph_runner.py` at 0.5.13.post1 (it only moves later in main).

## Changes applied in this fork
1. **Version-tolerant imports** (work on BOTH 0.5.12 and >=0.5.13):
   - `atom/plugin/sglang/models/deepseek_mla.py`
   - `atom/plugin/sglang/models/deepseek_mla_forward.py`
     ```python
     try:
         from sglang.srt.configs.model_config import is_deepseek_dsa as is_deepseek_nsa  # >=0.5.13
     except ImportError:
         from sglang.srt.configs.model_config import is_deepseek_nsa  # 0.5.12
     ```
   - `atom/plugin/sglang/models/deepseek_mla_attention.py`
     ```python
     try:
         from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp  # 0.5.12
     except ImportError:
         from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp as nsa_use_prefill_cp  # >=0.5.13
     ```
2. **docker/Dockerfile**
   - default `ARG SGLANG_REF="v0.5.13.post1"`
   - new step `[SGLANG-ATOM 4.5/6]` pins `huggingface_hub==1.20.1` + `transformers==5.8.1`
     before the 5/6 validate (fixes the `@strict`/Cohere2MoeConfig build crash).

## Build
```bash
docker build -f docker/Dockerfile --target atom_sglang \
  --build-arg SGLANG_BASE_IMAGE=rocm/atom-dev:latest \
  --build-arg SGLANG_REF=v0.5.13.post1 \
  --build-arg MAX_JOBS=$(nproc) \
  -t atom-sglang:v0.5.13.post1 .
```

## Remaining work — needs the container + GPU (import fixes != runtime correctness)
The 2 fixes above make ATOM's plugin *import* on 0.5.13.post1. Runtime still
needs validation because the NSA->DSA rework changed more than names:
1. `mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=None)` vs the old
   `nsa_use_prefill_cp(forward_batch, nsa_enable_prefill_cp=None)` — signature is
   compatible for the single positional call ATOM makes (`deepseek_mla_attention.py:185`),
   but confirm behaviour (single-node TP8 with no context-parallel -> returns False).
2. ATOM's `deepseek_mla_forward.py` / `deepseek_mla_attention.py` reimplement the
   MLA/indexer path against 0.5.12 internals; diff 0.5.12 `nsa_indexer.py` vs
   0.5.13 `dsa/dsa_indexer.py` for changed method signatures ATOM calls into.
3. Run the smoke test: launch GLM-5.2 NATIVE first (no ATOM) to confirm kernels,
   then with `SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models`.
4. If any further `ImportError`/`AttributeError` appears, extend the same
   try/except pattern (grep `from sglang` in `atom/plugin/sglang/`).

## Alternative: Direction A (no ATOM renovation)
See `scripts/direction_a_backport_sgl_kernel.sh` — rebuild ONLY sgl-kernel from
v0.5.13.post1 inside the 0.5.12 container, keep ATOM on 0.5.12. Smaller change;
risk is 0.5.13 kernel ABI vs 0.5.12 python op-schema mismatch.
