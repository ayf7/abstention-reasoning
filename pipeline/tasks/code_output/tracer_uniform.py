"""Execution-uniform tracing for code_output hint generation.

Unlike tracer.py which places checkpoints between top-level statements
(making loops opaque), this tracer distributes checkpoints evenly across
*execution units* — where each loop iteration counts as a separate unit.

Two-pass approach:
  1. Counting pass: instrument code with a lightweight global counter
     to measure total execution units. No state capture.
  2. Capture pass: knowing the total, compute 5 evenly-spaced checkpoint
     positions and instrument code to capture state only at those points.

This ensures hints are roughly uniformly distributed across the actual
computation, even for loop-heavy competitive programming code.
"""

import ast
import multiprocessing
import sys
import textwrap
from io import StringIO

# Names to exclude from variable snapshots
_INTERNAL_NAMES = frozenset({
    "_cp_", "_checkpoints_", "_copy_", "_snap_", "_k_", "_v_", "_n_", "_d_",
    "_step_", "_targets_", "_total_steps_",
    "__builtins__", "__name__", "__doc__",
    "__package__", "__loader__", "__spec__", "__file__",
})

MAX_REPR_LEN = 80


def trace_execution_uniform(
    code: str, stdin_input: str, num_hints: int = 5
) -> list[str]:
    """Execute code with stdin, capture variable state at ~num_hints
    evenly-spaced execution points.

    Returns list of hint strings like:
        "After step 42/200 (21%): var_1 = 5, var_2 = [2, 3, 1]"

    Returns empty list if tracing fails for any reason.

    Two-pass: first counts execution units, then captures state at
    evenly-spaced positions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    try:
        body, indent, in_function = _find_entry_body(tree, code)
    except ValueError:
        return []

    if len(body) < 2:
        return []

    # --- Pass 1: count total execution units ---
    try:
        counting_code = _instrument_counting(code, body, indent, in_function)
    except Exception:
        return []

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_counting_worker,
        args=(counting_code, stdin_input, result_queue),
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
        total_steps = result_queue.get_nowait()
    except Exception:
        return []

    if not isinstance(total_steps, int) or total_steps < 2:
        return []

    # Compute evenly-spaced checkpoint positions (1-indexed)
    targets = set()
    for i in range(1, num_hints + 1):
        pos = i * total_steps // (num_hints + 1)
        targets.add(pos)
    # Always include the final step
    targets.add(total_steps)
    # Cap at num_hints targets (keep the evenly-spaced ones + final)
    targets = sorted(targets)
    if len(targets) > num_hints:
        # Keep evenly spaced subset
        step = len(targets) / num_hints
        targets = [targets[int(i * step)] for i in range(num_hints)]
        targets = sorted(set(targets))

    # --- Pass 2: capture state at target positions ---
    try:
        capture_code = _instrument_capture(code, body, indent, targets, in_function)
    except Exception:
        return []

    result_queue2 = multiprocessing.Queue()
    proc2 = multiprocessing.Process(
        target=_capture_worker,
        args=(capture_code, stdin_input, result_queue2),
    )
    proc2.start()
    proc2.join(timeout=5)

    if proc2.is_alive():
        proc2.kill()
        proc2.join(timeout=2)
        return []

    if proc2.exitcode != 0:
        return []

    try:
        return result_queue2.get_nowait()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# AST helpers (shared with tracer.py, duplicated to keep modules independent)
# ---------------------------------------------------------------------------

def _find_entry_body(tree: ast.Module, source: str) -> tuple[list[ast.stmt], str, bool]:
    """Find the main body of statements to trace.

    Strategy:
    1. Look for `if __name__ == "__main__": func_name()` and trace that function
    2. Otherwise, if there are meaningful module-level statements, trace those
    3. Otherwise, find the biggest function and trace its body

    Returns (list of statement nodes, indentation string, in_function flag).
    Raises ValueError if no suitable body found.
    """
    module_body = tree.body
    if not module_body:
        raise ValueError("Empty module")

    entry_func_name = _find_main_call(module_body)

    if entry_func_name:
        for node in module_body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_func_name:
                return node.body, _get_indent(source, node.body[0]), True

    executable_stmts = [
        s for s in module_body
        if not isinstance(s, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                              ast.AsyncFunctionDef, ast.ClassDef))
    ]

    if len(executable_stmts) >= 2:
        return module_body, "", False

    func_defs = [s for s in module_body if isinstance(s, ast.FunctionDef)]
    if func_defs:
        biggest = max(func_defs, key=lambda f: f.end_lineno - f.lineno)
        return biggest.body, _get_indent(source, biggest.body[0]), True

    if len(module_body) >= 2:
        return module_body, "", False

    raise ValueError("Not enough statements to trace")


def _find_main_call(body: list[ast.stmt]) -> str | None:
    """Check for `if __name__ == "__main__": some_func()` and return function name."""
    for node in body:
        if not isinstance(node, ast.If):
            continue
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


# ---------------------------------------------------------------------------
# Pass 1: counting instrumentation
# ---------------------------------------------------------------------------

def _instrument_counting(source: str, body: list[ast.stmt], indent: str, in_function: bool) -> str:
    """Insert lightweight step counters into the code.

    Inserts `_step_ += 1` after each non-loop statement and at the end of
    each loop body (so each iteration counts as one step). Recurses into
    nested loops.
    """
    lines = source.splitlines()
    insertions: list[tuple[int, str, str]] = []  # (line_no, code, indent)

    _collect_counter_insertions(body, indent, insertions)

    # If entry body is inside a function, inject global declaration
    if in_function:
        first_line = body[0].lineno  # 1-indexed
        insertions.append((first_line - 1, "global _step_", indent))

    # Insert from bottom to top
    for line_no, code, ind in sorted(insertions, key=lambda x: x[0], reverse=True):
        lines.insert(line_no, ind + code)

    infra = ["_step_ = 0"]
    return "\n".join(infra + lines)


def _collect_counter_insertions(
    stmts: list[ast.stmt],
    indent: str,
    insertions: list[tuple[int, str, str]],
):
    """Recursively collect counter insertion points for a block of statements.

    Inserts `_step_ += 1` as the first statement in each loop body (counting
    each iteration as one execution unit) and after each non-loop top-level
    statement. Recurses into nested loops only.

    Inserting at the *start* of loop bodies (before the first statement)
    avoids indentation ambiguity at block boundaries.
    """
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.While)):
            loop_body = stmt.body
            if loop_body:
                loop_indent = " " * loop_body[0].col_offset
                # Insert counter at start of loop body (before first stmt)
                first_line = loop_body[0].lineno  # 1-indexed
                insertions.append((first_line - 1, "_step_ += 1", loop_indent))
                # Recurse into nested loops
                _collect_counter_insertions(loop_body, loop_indent, insertions)
        else:
            # Non-loop statement: count it by inserting before the next line
            # Use the statement's own start line to insert before it
            insertions.append((stmt.lineno - 1, "_step_ += 1", indent))


def _counting_worker(
    modified_code: str,
    stdin_input: str,
    result_queue: multiprocessing.Queue,
):
    """Subprocess worker: execute instrumented code, return total step count."""
    sys.stdin = StringIO(stdin_input)
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(modified_code, namespace)
    except Exception:
        result_queue.put(0)
        return

    result_queue.put(namespace.get("_step_", 0))


# ---------------------------------------------------------------------------
# Pass 2: capture instrumentation
# ---------------------------------------------------------------------------

def _instrument_capture(
    source: str,
    body: list[ast.stmt],
    indent: str,
    targets: list[int],
    in_function: bool,
) -> str:
    """Insert checkpoint captures at target step positions.

    Same step-counting logic as pass 1, but instead of just incrementing,
    checks if the current step is a target and captures state if so.
    """
    lines = source.splitlines()
    insertions: list[tuple[int, str, str]] = []  # (line_no, code, indent)

    _collect_capture_insertions(body, indent, insertions)

    # If entry body is inside a function, inject global declarations
    if in_function:
        first_line = body[0].lineno  # 1-indexed
        insertions.append((
            first_line - 1,
            "global _step_, _targets_, _checkpoints_, _cp_",
            indent,
        ))

    # Insert from bottom to top
    for line_no, code, ind in sorted(insertions, key=lambda x: x[0], reverse=True):
        lines.insert(line_no, ind + code)

    # Prepend infrastructure
    infra = [
        "import copy as _copy_",
        "_step_ = 0",
        f"_targets_ = set({targets!r})",
        "_checkpoints_ = []",
        "def _cp_(_d_):",
        "    _snap_ = {}",
        "    for _k_, _v_ in _d_.items():",
        "        try: _snap_[_k_] = _copy_.deepcopy(_v_)",
        "        except Exception: _snap_[_k_] = _v_",
        "    _checkpoints_.append((_step_, _snap_))",
    ]

    return "\n".join(infra + lines)


def _collect_capture_insertions(
    stmts: list[ast.stmt],
    indent: str,
    insertions: list[tuple[int, str, str]],
):
    """Recursively collect capture insertion points for a block of statements.

    Same strategy as _collect_counter_insertions: insert at start of loop
    bodies and before non-loop statements to avoid indentation issues.
    """
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.While)):
            loop_body = stmt.body
            if loop_body:
                loop_indent = " " * loop_body[0].col_offset
                first_line = loop_body[0].lineno
                capture_line = (
                    "_step_ += 1\n"
                    f"{loop_indent}if _step_ in _targets_: _cp_(dict(locals()))"
                )
                insertions.append((first_line - 1, capture_line, loop_indent))
                _collect_capture_insertions(loop_body, loop_indent, insertions)
        else:
            capture_line = (
                "_step_ += 1\n"
                f"{indent}if _step_ in _targets_: _cp_(dict(locals()))"
            )
            insertions.append((stmt.lineno - 1, capture_line, indent))


def _capture_worker(
    modified_code: str,
    stdin_input: str,
    result_queue: multiprocessing.Queue,
):
    """Subprocess worker: execute code with captures, format and return hints."""
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

    total_steps = namespace.get("_step_", 0)
    hints = _format_hints(checkpoints, total_steps)
    result_queue.put(hints)


# ---------------------------------------------------------------------------
# Hint formatting
# ---------------------------------------------------------------------------

def _format_hints(
    checkpoints: list[tuple[int, dict]],
    total_steps: int,
) -> list[str]:
    """Format checkpoint data into human-readable hint strings.

    Each hint shows the variable state at that execution point.
    Format: "After step 42/200: var_1 = 5, var_2 = [2, 3, 1]"
    """
    import types
    excluded_types = (type, types.ModuleType, types.FunctionType)

    hints = []

    for step_num, variables in checkpoints:
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
        pct = int(100 * step_num / total_steps) if total_steps > 0 else 0
        hints.append(f"After step {step_num}/{total_steps} ({pct}%): {vars_str}")

    return hints
