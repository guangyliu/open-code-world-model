"""Report names a function reads but never binds.

Three separate runs were lost to this exact fault: a patch replaced a block of `run_cp.py` and
silently took an assignment out with it -- `refs` twice, then `anchor` -- and each one only
surfaced as a NameError several minutes into a GPU allocation. `python -m ast` parses such a
file happily; nothing on the cluster lints. This is the smallest thing that would have caught
all three in under a second.

Deliberately conservative: it only flags a name that is read somewhere in a function and bound
nowhere in it, and that is not a module global, an import, a builtin, or a parameter. Closures
over enclosing functions are resolved, so a nested helper reading its parent's locals is fine.
"""
from __future__ import annotations

import ast
import builtins
import sys


def _bound(node: ast.AST) -> set[str]:
    """Every name this node binds, at its own level (not inside nested functions)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node:
            out.add(n.name)
            continue
        if isinstance(n, ast.ClassDef) and n is not node:
            out.add(n.name)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def _read(node: ast.AST) -> dict[str, int]:
    """Every name read, with the first line it is read on."""
    out: dict[str, int] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.setdefault(n.id, n.lineno)
    return out


def check(path: str) -> int:
    tree = ast.parse(open(path).read(), path)
    module_names = _bound(tree) | set(dir(builtins))
    problems: list[tuple[str, str, int]] = []

    def walk_fn(fn: ast.AST, enclosing: set[str]) -> None:
        here = _bound(fn) | enclosing
        for name, line in _read(fn).items():
            if name not in here:
                problems.append((getattr(fn, "name", "<module>"), name, line))
        for n in ast.iter_child_nodes(fn):
            for sub in ast.walk(n):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk_fn(sub, here)

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            walk_fn(n, module_names)
            break_outer = False
            for sub in ast.walk(n):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not n:
                    break_outer = True
            if break_outer:
                pass

    seen = set()
    uniq = []
    for fn, name, line in problems:
        if (fn, name) in seen:
            continue
        seen.add((fn, name))
        uniq.append((fn, name, line))

    if uniq:
        print(f"{path}: {len(uniq)} possibly-undefined name(s)")
        for fn, name, line in sorted(uniq, key=lambda t: t[2]):
            print(f"  line {line}: {name!r} is read in {fn}() but bound nowhere in it")
        return 1
    print(f"{path}: no undefined names")
    return 0


if __name__ == "__main__":
    sys.exit(max(check(p) for p in sys.argv[1:]))
