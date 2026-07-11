import ast, sys

PATH = sys.argv[1]
TARGETS = {"atom.models.minimax_m2","atom.models.minimax_m3","atom.models.qwen3_5",
           "atom.models.deepseek_v4","atom.models.qwen3_next","atom.models.kimi_k25"}

src = open(PATH).read()
if "[atom] skip atom.models." in src:
    print("already guarded:", PATH); raise SystemExit
lines = src.splitlines(keepends=True)
tree = ast.parse(src)

edits = []  # (start_idx, end_idx, new_text)
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module in TARGETS:
        indent = " " * node.col_offset
        stmt = "".join(lines[node.lineno-1:node.end_lineno])
        # bound names = asname or name
        bound = [a.asname or a.name for a in node.names]
        # re-indent original statement under try (add 4 spaces to each line)
        orig = stmt.rstrip("\n")
        orig_indented = "\n".join(indent + "    " + ln[node.col_offset:] if ln.strip() else ln
                                  for ln in orig.splitlines())
        none_assign = indent + "    " + " = ".join(bound) + " = None"
        block = (f"{indent}try:\n"
                 f"{orig_indented}\n"
                 f"{indent}except Exception as _e:\n"
                 f"{none_assign}\n"
                 f"{indent}    print(f'[atom] skip {node.module}: {{_e}}')\n")
        edits.append((node.lineno-1, node.end_lineno, block))

# apply from bottom up
for start, end, block in sorted(edits, reverse=True):
    lines[start:end] = [block]

out = "".join(lines)
# filter None from the registry dict, right before the first function def
marker = "\ndef _register_custom_attention_to_sglang"
filt = ("\n_ATOM_SUPPORTED_MODELS = {k: v for k, v in _ATOM_SUPPORTED_MODELS.items() "
        "if v is not None}\n")
assert marker in out
out = out.replace(marker, filt + marker, 1)

open(PATH, "w").write(out)
compile(out, PATH, "exec")   # will raise if invalid
print("patched + compiles OK:", PATH)
