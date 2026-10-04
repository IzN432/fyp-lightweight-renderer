"""Checks the hand-written type stubs (python/lr/__init__.pyi and engine.pyi) against the compiled module, so they can't
silently drift: every public class, method, property, enum value and function the module exports must be
in the stubs with the same parameter names, and the stubs must not describe anything the module lacks.

The reference is generated from the module itself by nanobind's stubgen.

Run via ctest (`python.stubs`), or directly:
    PYTHONPATH=build/python python tests/python/test_stubs.py
"""

import ast
import pathlib
import subprocess
import sys
import tempfile

STUBS = pathlib.Path(__file__).resolve().parents[2] / "python" / "lr" / "__init__.pyi"


def parameters(node):
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return tuple(n for n in names if n not in ("self", "cls"))


def public(name):
    return not name.startswith("_") or name == "__init__"


def surface(tree):
    """{qualified name: set of parameter tuples (one per overload), or None for non-callables}."""
    found: dict[str, set[tuple[str, ...]] | None] = {}

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef) and public(node.name):
                found.setdefault(prefix + node.name, None)
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public(node.name):
                key = prefix + node.name
                is_property = any(
                    (isinstance(d, ast.Name) and d.id == "property")
                    or (isinstance(d, ast.Attribute) and d.attr in ("setter", "getter"))
                    for d in node.decorator_list
                )
                if is_property:
                    found.setdefault(key, None)
                else:
                    entry = found.setdefault(key, set())
                    if entry is not None:
                        entry.add(parameters(node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and public(target.id):
                        found.setdefault(prefix + target.id, None)

    visit(tree.body, "")
    return found


def compare(module, stubs_path, reference_path):
    """Problems between one stub file and the stubgen reference for the module it describes."""
    reference = surface(ast.parse(reference_path.read_text(encoding="utf-8")))
    stubs = surface(ast.parse(stubs_path.read_text(encoding="utf-8")))

    def exists_at_runtime(name):
        # stubgen skips some real names (exception classes registered with nb::exception, members a class
        # inherits, like Enum.name), so names it lacks are checked against the live package instead.
        target = module
        for part in name.split("."):
            if not hasattr(target, part):
                return False
            target = getattr(target, part)
        return True

    problems = []
    for name, params in sorted(reference.items()):
        if name not in stubs:
            problems.append(f"{stubs_path.name}: missing from stubs: {name}")
        elif params and stubs[name] and params != stubs[name]:
            problems.append(f"{stubs_path.name}: {name}: module takes {sorted(params)}, stubs say {sorted(stubs[name])}")
    for name in sorted(stubs.keys() - reference.keys()):
        if not exists_at_runtime(name):
            problems.append(f"{stubs_path.name}: in stubs but not in the module: {name}")
    return len(reference), problems


def main():
    import lr
    import lr.engine

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp)
        # -r: one stub per (sub)module, as _lr/__init__.pyi and _lr/engine.pyi.
        subprocess.run(
            [sys.executable, "-m", "nanobind.stubgen", "-m", "lr._lr", "-r", "-O", str(out), "-q"], check=True
        )
        checked, problems = 0, []
        for module, stubs_path, reference_path in [
            (lr, STUBS, out / "_lr" / "__init__.pyi"),
            (lr.engine, STUBS.with_name("engine.pyi"), out / "_lr" / "engine.pyi"),
        ]:
            count, found = compare(module, stubs_path, reference_path)
            checked += count
            problems += found

    # lr/engine.py lists the native submodule's names explicitly; a new binding must be added there too.
    native = {name for name in dir(lr._lr.engine) if not name.startswith("_")}
    for name in sorted(native - set(lr.engine.__all__)):
        problems.append(f"lr/engine.py: native lr._lr.engine.{name} is not exported")

    for problem in problems:
        print(problem)
    print(f"{checked} public names checked, {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
