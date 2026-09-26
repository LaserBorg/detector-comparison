"""Child entry point for one isolated benchmark configuration.

Run as ``python -m yolo_bench.orchestrate._child --runtime ... --model ...``.

Kept as its own module (rather than ``python -c "..."``) so the child is a real
importable program: tracebacks show real file/line numbers, and the same
:func:`yolo_bench.orchestrate.run_one` is exercised in-process and out-of-process.
"""

from __future__ import annotations

import sys

from . import _child_main

if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1:]))