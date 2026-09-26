"""Unified CLI: python -m yolo_bench {export,bench,predict,check,compare,run,list} ...

The CLI is a *convenience* over the library. The same work is available
programmatically:

  * inference: ``from yolo_bench import Detector``
  * matrices:  ``from yolo_bench import orchestrate``

For notebook-driven work, prefer ``orchestrate`` over this CLI so results can be
inspected and plotted in the same process.
"""

from __future__ import annotations

import argparse
import sys

_SUBCOMMANDS = (
    "export",     # .pt -> .onnx -> .engine
    "check",      # correctness gate vs ultralytics .predict()
    "bench",      # one config
    "predict",    # inference only: draw detections, save annotated mp4
    "compare",    # a matrix -> csv + md
    "run",        # named experiment (alias for compare --experiment)
    "list",       # list runtimes / experiments
)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _parser().print_help()
        return 0 if argv else 0

    sub, rest = argv[0], argv[1:]

    if sub == "export":
        from .export import main as run
        return run(rest)
    if sub == "check":
        from .check import main as run
        return run(rest)
    if sub == "bench":
        from .bench import main as run
        return run(rest)
    if sub == "predict":
        from .predict import main as run
        return run(rest)
    if sub == "compare":
        from .compare import main as run
        return run(rest)
    if sub == "run":
        from .compare import main as run
        return run(["--experiment", *rest] if rest and not rest[0].startswith("-")
                   else rest)
    if sub == "list":
        from .runtimes import describe

        print(describe())
        print()
        from . import orchestrate

        for name in orchestrate.experiment_names():
            exp = orchestrate.EXPERIMENTS[name]
            print(f"{name:20s} {exp.description}")
        return 0

    _parser().print_help()
    return 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yolo_bench",
        description="Detector inference stack + benchmark harness",
        epilog="Inference API: from yolo_bench import Detector | "
               "Matrices: from yolo_bench import orchestrate",
    )
    p.add_argument("command", nargs="?", choices=_SUBCOMMANDS,
                   help="subcommand to run")
    return p


if __name__ == "__main__":
    raise SystemExit(main())