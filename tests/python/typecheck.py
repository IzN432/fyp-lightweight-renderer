"""Type-checks the examples and Python tests against lr's stubs (python/lr/__init__.pyi) with mypy, so the
stubs stay usable, not just complete (test_stubs.py checks completeness).

mypy is optional: without it this exits with 77, which ctest reports as skipped.

Run via ctest (`python.typecheck`), or directly from the repo root:
    python tests/python/typecheck.py
"""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    try:
        import mypy  # noqa: F401
    except ImportError:
        print("mypy not installed (pip install mypy) - skipping")
        sys.exit(77)

    # Resolve `import lr` to the source stubs, not the built package, and `import meshes` to the examples.
    env = dict(os.environ, MYPYPATH=os.pathsep.join([str(ROOT / "python"), str(ROOT / "examples" / "python")]))
    result = subprocess.run(
        [
            sys.executable, "-m", "mypy",
            "--ignore-missing-imports", "--check-untyped-defs", "--no-incremental",
            str(ROOT / "examples" / "python"), str(ROOT / "tests" / "python"),
        ],
        env=env,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
