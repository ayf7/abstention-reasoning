"""Execution tracing for code_output hint generation.

Inserts checkpoint calls between groups of top-level statements,
executes the modified code with mocked stdin, and captures variable
state at each checkpoint. The resulting hints describe cumulative
variable state after executing each segment of code.
"""

import ast
import multiprocessing
import sys
from io import StringIO


# Names to exclude from variable snapshots
_INTERNAL_NAMES = frozenset({
    "_cp_", "_checkpoints_", "_copy_", "_snap_", "_k_", "_v_", "_n_", "_d_",
    "__builtins__", "__name__", "__doc__",
    "__package__", "__loader__", "__spec__", "__file__",
})

MAX_REPR_LEN = 80


def trace_execution(code: str, stdin_input: str, num_hints: int = 5) -> list[str]:
    """Execute code with stdin, capture variable state at ~num_hints checkpoints.

    Returns list of hint strings like:
        "After lines 1-8: var_1 = 5, var_2 = [2, 3, 1]"

    Returns empty list if tracing fails for any reason.

    Runs execution in a subprocess to handle infinite loops and bare
    except: clauses that would swallow signal-based timeouts.
    """
    # Do AST analysis in the main process (fast, no safety concerns)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    try:
        body, indent = _find_entry_body(tree, code)
    except ValueError:
        return []

    if len(body) < 2:
        return []

    groups = _group_statements(body, num_groups=num_hints)
    if not groups:
        return []

    try:
        modified = _insert_checkpoints(code, groups, indent)
    except Exception:
        return []

    # Serialize group line ranges for the subprocess (AST nodes aren't picklable)
    group_ranges = []
    for group in groups:
        first_line = group[0].lineno
        last_line = max(s.end_lineno or s.lineno for s in group)
        group_ranges.append((first_line, last_line))

    # Run execution + formatting in subprocess
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_trace_worker,
        args=(modified, stdin_input, group_ranges, result_queue),
    )
    proc.start()
    proc.join(timeout=5)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
        return []

    if proc.exitcode != 0:
        return []

    try:
        return result_queue.get_nowait()
    except Exception:
        return []


def _trace_worker(
    modified_code: str,
    stdin_input: str,
    group_ranges: list[tuple[int, int]],
    result_queue: multiprocessing.Queue,
):
    """Subprocess worker: execute code, format hints, return strings via queue."""
    sys.stdin = StringIO(stdin_input)
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(modified_code, namespace)
    except Exception:
        result_queue.put([])
        return

    checkpoints = namespace.get("_checkpoints_", [])
    if not checkpoints:
        result_queue.put([])
        return

    # Format hints in-process (avoids pickling raw dicts with modules/functions)
    hints = _format_hints(checkpoints, group_ranges)
    result_queue.put(hints)


def _find_entry_body(tree: ast.Module, source: str) -> tuple[list[ast.stmt], str]:
    """Find the main body of statements to trace.

    Strategy:
    1. Look for `if __name__ == "__main__": func_name()` pattern and trace that function
    2. Otherwise, if there are meaningful module-level statements, trace those
    3. Otherwise, find the biggest function and trace its body

    Returns (list of statement nodes, indentation string for that level).

    Raises ValueError if no suitable body found.
    """
    module_body = tree.body
    if not module_body:
        raise ValueError("Empty module")

    # Check for if __name__ == "__main__" calling a function
    entry_func_name = _find_main_call(module_body)

    if entry_func_name:
        # Find that function's body
        for node in module_body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_func_name:
                return node.body, _get_indent(source, node.body[0])

    # Filter module body to non-import, non-function-def statements
    # to see if there are meaningful top-level statements
    executable_stmts = [
        s for s in module_body
        if not isinstance(s, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                              ast.AsyncFunctionDef, ast.ClassDef))
    ]

    # If there are meaningful module-level statements, trace those
    if len(executable_stmts) >= 2:
        return module_body, ""

    # If module is mostly function defs, find the "biggest" function
    func_defs = [
        s for s in module_body if isinstance(s, ast.FunctionDef)
    ]
    if func_defs:
        biggest = max(func_defs, key=lambda f: f.end_lineno - f.lineno)
        return biggest.body, _get_indent(source, biggest.body[0])

    # Fall back to full module body
    if len(module_body) >= 2:
        return module_body, ""

    raise ValueError("Not enough statements to trace")


def _find_main_call(body: list[ast.stmt]) -> str | None:
    """Check for `if __name__ == "__main__": some_func()` and return the function name."""
    for node in body:
        if not isinstance(node, ast.If):
            continue
        # Check: __name__ == "__main__"
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue

        left = test.left
        comparators = test.comparators
        if not (isinstance(left, ast.Name) and left.id == "__name__"
                and len(comparators) == 1
                and isinstance(comparators[0], ast.Constant)
                and comparators[0].value == "__main__"):
            continue

        # Look for a bare function call in the if body
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Name):
                    return func.id
    return None


def _get_indent(source: str, node: ast.stmt) -> str:
    """Get the indentation of a statement node."""
    lines = source.splitlines()
    if node.lineno <= len(lines):
        line = lines[node.lineno - 1]
        return line[: len(line) - len(line.lstrip())]
    return ""


def _group_statements(stmts: list[ast.stmt], num_groups: int = 5) -> list[list[ast.stmt]]:
    """Group statements into ~num_groups evenly by statement count.

    Each group gets roughly len(stmts) / num_groups statements.
    If there are fewer statements than groups, each statement becomes its own group.
    """
    if not stmts:
        return []

    num_groups = min(num_groups, len(stmts))
    if num_groups <= 0:
        return []

    groups = []
    n = len(stmts)
    for i in range(num_groups):
        start = i * n // num_groups
        end = (i + 1) * n // num_groups
        if start < end:
            groups.append(stmts[start:end])
    return groups


def _insert_checkpoints(source: str, groups: list[list[ast.stmt]], indent: str) -> str:
    """Insert checkpoint calls after each group of statements.

    Inserts `_cp_(N, dict(locals()))` after each group at the correct indentation.
    Also prepends checkpoint infrastructure code.
    """
    lines = source.splitlines()

    # Build insertion points: after the last line of each group
    insertions: list[tuple[int, str]] = []  # (line_number, code_to_insert)
    for i, group in enumerate(groups):
        last_stmt = group[-1]
        insert_after = (last_stmt.end_lineno or last_stmt.lineno)
        checkpoint_code = f"{indent}_cp_({i}, dict(locals()))"
        insertions.append((insert_after, checkpoint_code))

    # Insert from bottom to top to preserve line numbers
    for line_no, code in sorted(insertions, reverse=True):
        lines.insert(line_no, code)

    # Prepend checkpoint infrastructure
    # Uses deepcopy to snapshot mutable objects (lists, dicts) at checkpoint time
    infra = [
        "import copy as _copy_",
        "_checkpoints_ = []",
        "def _cp_(_n_, _d_):",
        "    _snap_ = {}",
        "    for _k_, _v_ in _d_.items():",
        "        try: _snap_[_k_] = _copy_.deepcopy(_v_)",
        "        except Exception: _snap_[_k_] = _v_",
        "    _checkpoints_.append((_n_, _snap_))",
    ]

    return "\n".join(infra + lines)


def _format_hints(
    checkpoints: list[tuple[int, dict]],
    group_ranges: list[tuple[int, int]],
) -> list[str]:
    """Format checkpoint data into human-readable hint strings.

    Each hint shows cumulative variable state after a group of lines.
    Format: "After lines 1-8: var_1 = 5, var_2 = [2, 3, 1]"

    Args:
        checkpoints: List of (checkpoint_index, locals_dict) tuples
        group_ranges: List of (first_line, last_line) tuples for each group
    """
    # Types to exclude — defined here since this runs in subprocess
    import types
    excluded_types = (type, types.ModuleType, types.FunctionType)

    hints = []

    for cp_idx, variables in checkpoints:
        if cp_idx >= len(group_ranges):
            continue

        first_line, last_line = group_ranges[cp_idx]

        # Filter variables
        filtered = {}
        for name, value in sorted(variables.items()):
            if name.startswith("_") and (name.endswith("_") or name in _INTERNAL_NAMES):
                continue
            if name in _INTERNAL_NAMES:
                continue
            if isinstance(value, excluded_types):
                continue
            if callable(value):
                continue

            # Truncate long reprs
            try:
                r = repr(value)
            except Exception:
                continue

            if len(r) > MAX_REPR_LEN:
                r = r[:MAX_REPR_LEN - 3] + "..."

            filtered[name] = r

        if not filtered:
            continue

        vars_str = ", ".join(f"{name} = {val}" for name, val in filtered.items())

        if first_line == last_line:
            hints.append(f"After line {first_line}: {vars_str}")
        else:
            hints.append(f"After lines {first_line}-{last_line}: {vars_str}")

    return hints
