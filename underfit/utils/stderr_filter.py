"""Filter MLIR / LLVM Triton-compile noise from a subprocess pipeline.

When `flex_attention`'s Triton kernel fails to compile (typically on pre-
Ampere GPUs like T4), the LLVM backend dumps the entire intermediate
representation to stderr — hundreds of lines of `#blocked = ...`,
`tt.func @triton_...`, MLIR pass output, etc. Python's logging module can't
catch this because it's C++ writing directly to fd 2.

This script sits between the spawned training process and `tee`/the dashboard
log:

    python lora_train.py ... 2>&1 | python -u -m underfit.utils.stderr_filter | tee log

Reads stdin line-by-line, drops lines that match known compile-dump patterns,
forwards the rest with line-buffered output. The filter is intentionally
narrow: it matches only patterns specific to MLIR/LLVM output, never normal
Python tracebacks or tqdm progress bars.
"""
import re
import sys


# Lines matching this pattern are dropped. The alternatives are carefully
# scoped to MLIR / LLVM output. Bench against `train/loss=` lines, tqdm
# bars, and normal Python tracebacks — none of those should match.
NOISE_RE = re.compile(
    r"^("
    r"LLVM ERROR:"                                  # top of an LLVM crash banner
    r"|Unsupported conversion"                      # specific f16/f16 conversion error
    r"|#(blocked|shared|smem|loc)\d*\s*="           # MLIR attribute defs (#blocked = ...)
    r"|module attributes\s*\{"                      # MLIR top-level module
    r"|\s+tt\.|\s+ttg\.|\s+arith\.|\s+scf\."        # indented MLIR op uses
    r"|\s+nvgpu\.|\s+memref\.|\s+cf\.|\s+llvm\."
    r"|\s+%[\w]+\s*="                               # MLIR SSA values: %42 = …, %cst_3 = …, %c24_i32 = …
    r"|\s*\^bb\d+:?"                                # MLIR block labels (^bb0, ^bb1:)
    r"|\s*\}\s*\)*\s*$"                             # closing braces alone on a line
    r"|\s*\{\s*$"                                   # opening braces alone on a line
    r"|\s*tt\.func\s+"                              # tt.func declaration
    r"|\s*tt\.return\s*$"                           # tt.return alone
    r"|\s+mlir_reproducer:"                         # MLIR reproducer dump field
    r"|\s+pipeline:\s*\""                           # pipeline JSON inside reproducer
    r"|\s+disable_threading:"
    r"|\s+verify_each:"
    r"|/tmp/torchinductor_[^/]+/.*:\d+:\d+:"        # MLIR error location lines
    r"|#-\}"                                        # MLIR comment closer
    # ── Inductor autotuner output ──────────────────────────────────────
    # Fires every time a new attention shape is seen. Verbose, line-noise,
    # and the cached winner is what actually runs — autotune lines have
    # no diagnostic value for the user during normal training.
    r"|AUTOTUNE\s+\w+\("                            # banner: 'AUTOTUNE flex_attention(...)'
    r"|\s+triton_\w+_\d+\s+[\d.]+\s+ms\s+"          # candidate row: '  triton_flex_attention_N 0.53 ms 100.0% ...'
    r"|SingleProcess\s+AUTOTUNE\s+"                 # footer: 'SingleProcess AUTOTUNE benchmarking takes ...'
    r")"
)


def main():
    write = sys.stdout.write
    flush = sys.stdout.flush
    for line in iter(sys.stdin.readline, ""):
        if NOISE_RE.match(line):
            continue
        write(line)
        flush()


if __name__ == "__main__":
    main()
