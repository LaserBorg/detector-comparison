"""Unified CLI: python -m yolo_bench {export,bench,compare,check} ..."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _parser().print_help()
        return 0

    sub = argv[0]
    rest = argv[1:]

    if sub == "export":
        from .export import main as run
        return run(rest)
    if sub == "bench":
        from .bench import main as run
        return run(rest)
    if sub == "compare":
        from .compare import main as run
        return run(rest)
    if sub == "check":
        from .check import main as run
        return run(rest)

    _parser().print_help()
    return 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yolo_bench",
        description="YOLO11/26 -> ONNX -> TensorRT FP16/FP32 benchmark",
    )
    p.add_argument("command", nargs="?",
                   choices=["export", "bench", "compare", "check"],
                   help="subcommand to run")
    return p


if __name__ == "__main__":
    raise SystemExit(main())