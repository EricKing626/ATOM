#!/usr/bin/env bash
# =============================================================================
# check_sglang_compat.sh — verify ATOM's sglang-internal imports resolve against
# the sglang currently installed. Run INSIDE the target container.
#
# AST-based: understands try/except. It reports
#   [BREAK]  an UNGUARDED import that does not resolve  -> must fix
#   [note]   a guarded (try/except) import that does not resolve -> fallback exists, OK
# Exit 0 = no unguarded breaks.
#
# Usage:  bash scripts/check_sglang_compat.sh
#         PLUGIN_DIR=atom/plugin/sglang bash scripts/check_sglang_compat.sh
# NOTE: import-level only; runtime signature/behaviour still needs a launch test.
# =============================================================================
set -uo pipefail
PLUGIN_DIR="${PLUGIN_DIR:-atom/plugin/sglang}"
PY="${VENV_PYTHON:-python3}"

"$PY" - "$PLUGIN_DIR" <<'PY'
import ast, os, sys, importlib

plugin_dir = sys.argv[1]
try:
    import sglang
    print(f"sglang: {sglang.__version__} -> {sglang.__file__}")
except Exception as e:
    print("sglang NOT INSTALLED:", e); sys.exit(2)

def resolves(module, name):
    """name=None -> just import module; else module must expose attr name."""
    try:
        m = importlib.import_module(module)
    except Exception:
        return False
    if name is None:
        return True
    if hasattr(m, name):
        return True
    # attr may itself be a submodule
    try:
        importlib.import_module(module + "." + name); return True
    except Exception:
        return False

def guarded(node, ancestors):
    # True if this import node is inside a Try block's `body` (has an except)
    return any(isinstance(a, ast.Try) for a in ancestors)

breaks, notes, ok = [], [], 0

class V(ast.NodeVisitor):
    def __init__(self): self.stack = []
    def generic_visit(self, node):
        self.stack.append(node); super().generic_visit(node); self.stack.pop()
    def _check(self, module, name, node):
        global ok
        if not (module and module.startswith("sglang")): return
        g = guarded(node, self.stack)
        if resolves(module, name):
            ok += 1
        else:
            tgt = f"from {module} import {name}" if name else f"import {module}"
            (notes if g else breaks).append(tgt)
    def visit_ImportFrom(self, node):
        for a in node.names:
            self._check(node.module, a.name, node)
        self.generic_visit(node)
    def visit_Import(self, node):
        for a in node.names:
            self._check(a.name, None, node)
        self.generic_visit(node)

for root,_,files in os.walk(plugin_dir):
    for f in files:
        if f.endswith(".py"):
            p=os.path.join(root,f)
            try: tree=ast.parse(open(p).read(), p)
            except SyntaxError as e: print("skip (syntax)", p, e); continue
            V().visit(tree)

breaks=sorted(set(breaks)); notes=sorted(set(notes))
for b in breaks: print("[BREAK]", b)
for n in notes:  print("[note ]", n, "(guarded by try/except — fallback exists)")
print(f"== resolved: {ok}   unguarded breaks: {len(breaks)}   guarded-missing: {len(notes)} ==")
if breaks:
    print("Add a version-tolerant try/except for each [BREAK] above.")
    sys.exit(1)
print("NO UNGUARDED BREAKS. (Runtime behaviour still needs a launch smoke test.)")
PY
