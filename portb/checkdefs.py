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


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SCOPED = _NESTED + (ast.Lambda,)

# Names Python puts in every module scope that no statement binds.
_MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                   "__loader__", "__builtins__"}


def _record(n: ast.AST, out: set[str]) -> None:
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


def _own_body(node: ast.AST) -> list:
    b = getattr(node, "body", [])
    return b if isinstance(b, list) else [b]


def _descend(node: ast.AST, visit) -> None:
    """Apply `visit` to this scope's nodes, stopping at every nested definition.

    `ast.walk` does not stop, and `continue` inside a walk loop does not prune -- which is
    the whole defect: _bound(module) collected every local in the file, so the effective
    rule became "bound nowhere in the file" and a name assigned in one function and read
    in a sibling passed as clean while raising NameError at runtime.
    """
    def rec(n: ast.AST) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, _SCOPED):
                visit(child, nested=True)
                continue
            visit(child, nested=False)
            rec(child)

    for st in _own_body(node):
        if isinstance(st, _SCOPED):
            visit(st, nested=True)
            continue
        visit(st, nested=False)
        rec(st)


def _bound(node: ast.AST) -> set[str]:
    """Every name this node binds at its OWN level, never inside a nested definition."""
    out: set[str] = set()
    # Only callables have parameters; a ClassDef is a scope but has no `.args`.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for a in ast.walk(node.args):
            _record(a, out)

    def visit(n: ast.AST, nested: bool) -> None:
        if nested:
            if not isinstance(n, ast.Lambda):
                out.add(n.name)      # the definition binds its own name here
            return
        _record(n, out)

    _descend(node, visit)
    return out


def _read(node: ast.AST) -> dict[str, int]:
    """Every name read at this node's own level, with the first line it is read on.

    Also stops at nested definitions: a name read only inside a nested function belongs to
    that function's scope and is checked there, with its own enclosing set. Charging the
    parent with it would flag the child's locals as undefined in the parent.
    """
    out: dict[str, int] = {}

    def visit(n: ast.AST, nested: bool) -> None:
        if nested:
            # Decorators and default values ARE evaluated in the enclosing scope.
            for d in getattr(n, "decorator_list", []):
                for sub in ast.walk(d):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        out.setdefault(sub.id, sub.lineno)
            args = getattr(n, "args", None)
            for d in (list(getattr(args, "defaults", [])) +
                      [x for x in getattr(args, "kw_defaults", []) if x is not None]):
                for sub in ast.walk(d):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        out.setdefault(sub.id, sub.lineno)
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.setdefault(n.id, n.lineno)

    _descend(node, visit)
    return out


def _child_defs(node: ast.AST) -> list:
    """Definitions directly inside this scope, not those inside a deeper one."""
    out: list = []

    def visit(n: ast.AST, nested: bool) -> None:
        if nested and not isinstance(n, ast.Lambda):
            out.append(n)

    _descend(node, visit)
    return out


def check(path: str) -> int:
    tree = ast.parse(open(path).read(), path)
    module_names = _bound(tree) | set(dir(builtins)) | _MODULE_DUNDERS
    problems: list[tuple[str, str, int]] = []

    def walk_fn(fn: ast.AST, enclosing: set[str]) -> None:
        here = _bound(fn) | enclosing
        for name, line in _read(fn).items():
            if name not in here:
                problems.append((getattr(fn, "name", "<module>"), name, line))
        # A class body is NOT in the lookup chain of its own methods: `FIELD = 1` in the
        # body followed by `return FIELD` in a method is a runtime NameError. So methods
        # inherit the scope AROUND the class, plus the class name, and not its body.
        inner = (enclosing | {fn.name}) if isinstance(fn, ast.ClassDef) else here
        for sub in _child_defs(fn):
            walk_fn(sub, inner)

    for top in _child_defs(tree):
        walk_fn(top, module_names)

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
