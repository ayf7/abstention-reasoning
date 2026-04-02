"""Strategic execution tracing for code_output hint generation.

Produces two strategically-placed hints:
1. State after the first loop completion past 50% of execution
2. State right before the final print/output statement

Two-pass approach:
  1. Counting pass: count execution steps, identify post-loop completion
     points and print statement locations.
  2. Capture pass: capture variable state at the two chosen points.
"""

import ast
import multiprocessing
import sys
from io import StringIO

# Names to exclude from variable snapshots
_INTERNAL_NAMES = frozenset({
    "_cp_", "_checkpoints_", "_copy_", "_snap_", "_k_", "_v_", "_n_", "_d_",
    "_step_", "_targets_", "_total_steps_",
    "_post_loop_", "_print_steps_", "_in_loop_",
    "__builtins__", "__name__", "__doc__",
    "__package__", "__loader__", "__spec__", "__file__",
})

MAX_REPR_LEN = 80


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_true_steps(code: str, stdin_input: str, timeout: float = 5) -> int | None:
    """Count total line executions using sys.settrace.

    Counts every 'line' event including function calls, recursion, etc.
    Returns the count, or None on failure/timeout.
    """
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_true_steps_worker,
        args=(code, stdin_input, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
        return None

    if proc.exitcode != 0:
        return None

    try:
        return result_queue.get_nowait()
    except Exception:
        return None


def _true_steps_worker(
    code: str,
    stdin_input: str,
    result_queue: multiprocessing.Queue,
):
    """Subprocess worker: run code with sys.settrace to count line executions."""
    sys.stdin = StringIO(stdin_input)
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == 'line':
            count += 1
        return tracer

    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    try:
        compiled = compile(code, "<string>", "exec")
        sys.settrace(tracer)
        exec(compiled, namespace)
        sys.settrace(None)
    except Exception:
        sys.settrace(None)
        result_queue.put(None)
        return

    result_queue.put(count)


def trace_execution_uniform(
    code: str, stdin_input: str, num_hints: int = 2
) -> list[str]:
    """Execute code with stdin, capture variable state at two strategic points.

    Hint 1: after the first loop completion past 50% of execution.
    Hint 2: right before the final print() call.

    Returns list of hint strings. Returns empty list on failure.
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

    # --- Pass 1: count steps + metadata ---
    metadata = _run_counting_pass(code, body, indent, in_function, stdin_input)
    if metadata is None:
        return []

    total_steps = metadata["total_steps"]
    if total_steps < 2:
        return []

    choice = _choose_targets(
        total_steps,
        sorted(metadata.get("post_loop_steps", [])),
        sorted(metadata.get("print_steps", [])),
    )
    targets = choice["targets"]
    if not targets:
        return []

    # --- Pass 2: capture state at targets ---
    try:
        capture_code = _instrument_capture(code, body, indent, targets, in_function)
    except Exception:
        return []

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_capture_worker,
        args=(capture_code, stdin_input, result_queue),
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


def profile_execution(code: str, stdin_input: str) -> dict | None:
    """Run counting pass only, return execution metadata for profiling.

    Returns dict with hint placement info, loop dominance stats, etc.
    Returns None on failure.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    try:
        body, indent, in_function = _find_entry_body(tree, code)
    except ValueError:
        return None

    if len(body) < 2:
        return None

    num_top_loops = sum(1 for s in body if isinstance(s, (ast.For, ast.While)))

    metadata = _run_counting_pass(code, body, indent, in_function, stdin_input)
    if metadata is None:
        return None

    total_steps = metadata["total_steps"]
    if total_steps < 2:
        return None

    post_loop_steps = sorted(metadata.get("post_loop_steps", []))
    print_steps = sorted(metadata.get("print_steps", []))
    in_loop_steps = metadata.get("in_loop_steps", 0)

    choice = _choose_targets(total_steps, post_loop_steps, print_steps)

    return {
        "total_steps": total_steps,
        "in_loop_steps": in_loop_steps,
        "loop_fraction": 100 * in_loop_steps / total_steps,
        "num_top_loops": num_top_loops,
        "num_post_loop_points": len(set(post_loop_steps)),
        "num_print_steps": len(set(print_steps)),
        **choice,
    }


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def _choose_targets(
    total_steps: int,
    post_loop_steps: list[int],
    print_steps: list[int],
) -> dict:
    """Choose the two hint target step numbers.

    Hint 1: first post-loop-completion step >= 50% of total execution.
             Falls back to ceil(50%) if no qualifying post-loop step exists.
    Hint 2: step of the last print() execution (state captured before it runs).
             Falls back to total_steps if no print detected.

    Returns dict with targets list and metadata for profiling.
    """
    midpoint = total_steps * 0.5

    # Hint 1: first post-loop step >= 50%
    hint1 = None
    hint1_is_post_loop = False
    for s in post_loop_steps:
        if s >= midpoint:
            hint1 = s
            hint1_is_post_loop = True
            break
    if hint1 is None:
        hint1 = min(int(midpoint) + 1, total_steps)

    # Hint 2: last print execution
    hint2_is_print = False
    if print_steps:
        hint2 = print_steps[-1]
        hint2_is_print = True
    else:
        hint2 = total_steps

    targets = sorted(set([hint1, hint2]))

    return {
        "targets": targets,
        "hint1_step": hint1,
        "hint1_pct": 100 * hint1 / total_steps,
        "hint1_is_post_loop": hint1_is_post_loop,
        "hint2_step": hint2,
        "hint2_pct": 100 * hint2 / total_steps,
        "hint2_is_print": hint2_is_print,
    }


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _find_entry_body(
    tree: ast.Module, source: str
) -> tuple[list[ast.stmt], str, bool]:
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


def _is_print_call(stmt: ast.stmt) -> bool:
    """Check if a statement is a bare print() call."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        if isinstance(func, ast.Name) and func.id == "print":
            return True
    return False


# ---------------------------------------------------------------------------
# Pass 1: counting instrumentation
# ---------------------------------------------------------------------------

def _run_counting_pass(
    code: str,
    body: list[ast.stmt],
    indent: str,
    in_function: bool,
    stdin_input: str,
) -> dict | None:
    """Run the counting pass in a subprocess, return metadata dict or None."""
    try:
        counting_code = _instrument_counting(code, body, indent, in_function)
    except Exception:
        return None

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
        return None

    if proc.exitcode != 0:
        return None

    try:
        metadata = result_queue.get_nowait()
    except Exception:
        return None

    if not isinstance(metadata, dict) or "total_steps" not in metadata:
        return None

    return metadata


def _instrument_counting(
    source: str,
    body: list[ast.stmt],
    indent: str,
    in_function: bool,
) -> str:
    """Insert step counters + metadata trackers into the code.

    Tracks:
    - _step_: total execution steps
    - _post_loop_: set of step numbers immediately after a loop completes
                   (entry-body level only)
    - _print_steps_: set of step numbers that are print() calls
    - _in_loop_: count of steps executed inside any loop body
    """
    lines = source.splitlines()
    insertions: list[tuple[int, str, str]] = []

    _collect_counter_insertions(body, indent, insertions, in_loop=False)

    if in_function:
        first_line = body[0].lineno
        insertions.append((
            first_line - 1,
            "global _step_, _post_loop_, _print_steps_, _in_loop_",
            indent,
        ))

    for line_no, code, ind in sorted(insertions, key=lambda x: x[0], reverse=True):
        lines.insert(line_no, ind + code)

    infra = [
        "_step_ = 0",
        "_post_loop_ = set()",
        "_print_steps_ = set()",
        "_in_loop_ = 0",
    ]
    return "\n".join(infra + lines)


def _collect_counter_insertions(
    stmts: list[ast.stmt],
    indent: str,
    insertions: list[tuple[int, str, str]],
    in_loop: bool = False,
):
    """Recursively collect counter + metadata insertion points.

    Each non-loop statement gets `_step_ += 1`. Additionally:
    - Inside loops: also increments `_in_loop_`
    - After a loop at entry-body level (in_loop=False): records post-loop step
    - Print calls: records print step
    """
    prev_was_loop = False
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.While)):
            loop_body = stmt.body
            if loop_body:
                loop_indent = " " * loop_body[0].col_offset
                _collect_counter_insertions(
                    loop_body, loop_indent, insertions, in_loop=True,
                )
            prev_was_loop = True
        else:
            code = "_step_ += 1"
            if in_loop:
                code += f"\n{indent}_in_loop_ += 1"
            if prev_was_loop and not in_loop:
                code += f"\n{indent}_post_loop_.add(_step_)"
            if _is_print_call(stmt):
                code += f"\n{indent}_print_steps_.add(_step_)"
            insertions.append((stmt.lineno - 1, code, indent))
            prev_was_loop = False


def _counting_worker(
    modified_code: str,
    stdin_input: str,
    result_queue: multiprocessing.Queue,
):
    """Subprocess worker: execute instrumented code, return metadata dict."""
    sys.stdin = StringIO(stdin_input)
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(modified_code, namespace)
    except Exception:
        result_queue.put({})
        return

    result_queue.put({
        "total_steps": namespace.get("_step_", 0),
        "post_loop_steps": sorted(namespace.get("_post_loop_", set())),
        "print_steps": sorted(namespace.get("_print_steps_", set())),
        "in_loop_steps": namespace.get("_in_loop_", 0),
    })


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

    Same step-counting logic as pass 1 (recurse into loops, count non-loop
    statements), but checks if the current step is a target and captures
    state if so.
    """
    lines = source.splitlines()
    insertions: list[tuple[int, str, str]] = []

    _collect_capture_insertions(body, indent, insertions)

    if in_function:
        first_line = body[0].lineno
        insertions.append((
            first_line - 1,
            "global _step_, _targets_, _checkpoints_, _cp_",
            indent,
        ))

    for line_no, code, ind in sorted(insertions, key=lambda x: x[0], reverse=True):
        lines.insert(line_no, ind + code)

    infra = [
        "import copy as _copy_",
        "_step_ = 0",
        f"_targets_ = set({targets!r})",
        "_checkpoints_ = []",
        "def _cp_(_d_, _line_):",
        "    _snap_ = {}",
        "    for _k_, _v_ in _d_.items():",
        "        try: _snap_[_k_] = _copy_.deepcopy(_v_)",
        "        except Exception: _snap_[_k_] = _v_",
        "    _checkpoints_.append((_step_, _line_, _snap_))",
    ]

    return "\n".join(infra + lines)


def _collect_capture_insertions(
    stmts: list[ast.stmt],
    indent: str,
    insertions: list[tuple[int, str, str]],
):
    """Recursively collect capture insertion points.

    Must match the same recursion/counting pattern as _collect_counter_insertions
    so that step numbers are identical between the two passes.
    """
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.While)):
            loop_body = stmt.body
            if loop_body:
                loop_indent = " " * loop_body[0].col_offset
                _collect_capture_insertions(loop_body, loop_indent, insertions)
        else:
            capture_line = (
                "_step_ += 1\n"
                f"{indent}if _step_ in _targets_: _cp_(dict(locals()), {stmt.lineno})"
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
    checkpoints: list[tuple[int, int, dict]],
    total_steps: int,
) -> list[str]:
    """Format checkpoint data into human-readable hint strings.

    Format: "After step 42/200 (21%, line 15): var_1 = 5, var_2 = [2, 3, 1]"
    """
    import types
    excluded_types = (type, types.ModuleType, types.FunctionType)

    hints = []

    for step_num, line_num, variables in checkpoints:
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
        hints.append(f"After step {step_num}/{total_steps} ({pct}%, line {line_num}): {vars_str}")

    return hints
