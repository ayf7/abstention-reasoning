"""AST-based identifier anonymization for Python code.

Renames user-defined identifiers (functions, classes, parameters, variables)
to generic names while preserving string literals and code semantics.
"""

import ast
import builtins
import keyword
import re


# Names that should never be renamed
PROTECTED_NAMES = (
    set(dir(builtins))
    | set(keyword.kwlist)
    | {"self", "cls", "super", "property", "staticmethod", "classmethod",
       "print", "len", "range", "enumerate", "zip", "map", "filter",
       "sorted", "reversed", "min", "max", "sum", "abs", "all", "any",
       "isinstance", "issubclass", "type", "object", "str", "int", "float",
       "bool", "list", "dict", "set", "tuple", "frozenset", "bytes",
       "bytearray", "memoryview", "complex", "slice", "iter", "next",
       "input", "open", "chr", "ord", "hex", "oct", "bin",
       "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
       "StopIteration", "Exception", "IOError", "OSError", "AttributeError",
       "ImportError", "NameError", "ZeroDivisionError", "OverflowError",
       "None", "True", "False", "NotImplemented", "Ellipsis",
       "math", "collections", "itertools", "functools", "operator",
       "sys", "os", "re", "string", "copy", "heapq", "bisect", "random",
       "defaultdict", "Counter", "deque", "OrderedDict", "namedtuple",
       "lru_cache", "reduce", "partial", "wraps",
       "inf", "nan", "pi", "e",
       "stdin", "stdout", "stderr",
       "append", "extend", "insert", "remove", "pop", "clear",
       "index", "count", "sort", "reverse", "copy",
       "keys", "values", "items", "get", "update", "setdefault",
       "add", "discard", "union", "intersection", "difference",
       "join", "split", "strip", "lstrip", "rstrip", "replace",
       "startswith", "endswith", "find", "rfind", "upper", "lower",
       "isdigit", "isalpha", "isalnum", "isupper", "islower",
       "format", "encode", "decode",
       }
)


class _IdentifierCollector(ast.NodeVisitor):
    """Collect user-defined identifiers from an AST."""

    def __init__(self, fn_name: str | None = None):
        self.fn_name = fn_name
        self.functions: list[str] = []    # function defs (excluding fn_name)
        self.classes: list[str] = []      # class defs
        self.params: list[str] = []       # function parameters
        self.variables: list[str] = []    # assignment targets, loop vars

        self._seen: set[str] = set()

    def _should_collect(self, name: str) -> bool:
        """Check if name should be collected for renaming."""
        if name in PROTECTED_NAMES:
            return False
        if name.startswith("__") and name.endswith("__"):
            return False
        if name in self._seen:
            return False
        return True

    def _add(self, name: str, category: list[str]):
        if self._should_collect(name):
            self._seen.add(name)
            category.append(name)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.fn_name is not None and node.name == self.fn_name:
            pass  # Entry function collected separately
        else:
            self._add(node.name, self.functions)
        # Collect parameters
        for arg in node.args.args:
            self._add(arg.arg, self.params)
        for arg in node.args.posonlyargs:
            self._add(arg.arg, self.params)
        for arg in node.args.kwonlyargs:
            self._add(arg.arg, self.params)
        if node.args.vararg:
            self._add(node.args.vararg.arg, self.params)
        if node.args.kwarg:
            self._add(node.args.kwarg.arg, self.params)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        self._add(node.name, self.classes)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self._add(node.id, self.variables)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self._collect_targets(node.target)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        self._collect_targets(node.target)
        self.generic_visit(node)

    def _collect_targets(self, target: ast.AST):
        """Recursively collect assignment targets (handles tuple unpacking)."""
        if isinstance(target, ast.Name):
            self._add(target.id, self.variables)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_targets(elt)
        elif isinstance(target, ast.Starred):
            self._collect_targets(target.value)


def _find_string_spans(source: str) -> list[tuple[int, int]]:
    """Find protected string literal spans in source code.

    For regular strings, the entire span is protected from renaming.
    For f-strings, only the literal text is protected — {expression}
    regions are left unprotected so identifiers inside them get renamed.
    """
    spans = []
    # Match triple-quoted first (greedy), then single-quoted
    pattern = re.compile(
        r'[fFrRbBuU]*"""[\s\S]*?"""|'
        r"[fFrRbBuU]*'''[\s\S]*?'''|"
        r'[fFrRbBuU]*"(?:[^"\\]|\\.)*"|'
        r"[fFrRbBuU]*'(?:[^'\\]|\\.)*'",
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        matched = m.group()
        # Determine prefix (f, r, b, etc.)
        prefix_end = next(i for i, c in enumerate(matched) if c in ('"', "'"))
        prefix = matched[:prefix_end].lower()

        if 'f' in prefix:
            spans.extend(_fstring_literal_spans(matched, m.start()))
        else:
            spans.append((m.start(), m.end()))
    return spans


def _fstring_literal_spans(text: str, offset: int) -> list[tuple[int, int]]:
    """Return protected (literal-only) spans for an f-string.

    Expressions inside {...} are excluded so identifiers get renamed.
    """
    quote_pos = next(i for i, c in enumerate(text) if c in ('"', "'"))
    quote_char = text[quote_pos]
    delim_len = 3 if text[quote_pos:quote_pos + 3] == quote_char * 3 else 1

    content_start = quote_pos + delim_len
    content_end = len(text) - delim_len

    spans = []
    lit_start = 0  # start of current literal region (text-relative)
    pos = content_start

    while pos < content_end:
        ch = text[pos]
        if ch == '{' and pos + 1 < content_end and text[pos + 1] == '{':
            pos += 2  # escaped {{ — stays literal
        elif ch == '}' and pos + 1 < content_end and text[pos + 1] == '}':
            pos += 2  # escaped }}
        elif ch == '{':
            # Protect literal text before this expression
            if pos > lit_start:
                spans.append((offset + lit_start, offset + pos))
            # Find matching }
            depth = 1
            pos += 1
            while pos < content_end and depth > 0:
                c = text[pos]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                elif c in ('"', "'"):
                    # Skip nested string inside expression
                    q = c
                    pos += 1
                    while pos < content_end and text[pos] != q:
                        if text[pos] == '\\':
                            pos += 1
                        pos += 1
                pos += 1
            lit_start = pos  # resume literal after '}'
        else:
            pos += 1

    # Final literal span including closing delimiter
    if len(text) > lit_start:
        spans.append((offset + lit_start, offset + len(text)))

    return spans


def _apply_renames(source: str, rename_map: dict[str, str]) -> str:
    """Apply renames to source, protecting string literals.

    Strategy: split source into string and non-string segments,
    apply word-boundary replacements only to non-string segments.
    """
    if not rename_map:
        return source

    string_spans = _find_string_spans(source)

    # Build segments: (start, end, is_string)
    segments = []
    pos = 0
    for span_start, span_end in sorted(string_spans):
        if span_start > pos:
            segments.append((pos, span_start, False))
        segments.append((span_start, span_end, True))
        pos = span_end
    if pos < len(source):
        segments.append((pos, len(source), False))

    # Sort by longest name first to avoid partial replacements
    sorted_names = sorted(rename_map.keys(), key=len, reverse=True)

    # Build combined pattern
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(name) for name in sorted_names) + r')\b'
    )

    result_parts = []
    for start, end, is_string in segments:
        text = source[start:end]
        if is_string:
            result_parts.append(text)
        else:
            text = pattern.sub(lambda m: rename_map[m.group(0)], text)
            result_parts.append(text)

    return ''.join(result_parts)


def _strip_comments(source: str) -> str:
    """Remove comments from source code, preserving string literals.

    Removes:
    - Full-line comments (lines that are only a comment, possibly indented)
    - Inline comments (# ... at end of a code line)
    Does NOT touch # inside string literals.
    """
    string_spans = _find_string_spans(source)

    def _in_string(pos: int) -> bool:
        for start, end in string_spans:
            if start <= pos < end:
                return True
        return False

    lines = source.split('\n')
    result = []
    pos = 0  # track position in original source

    for line in lines:
        # Find first # not inside a string
        comment_pos = None
        for j, ch in enumerate(line):
            if ch == '#' and not _in_string(pos + j):
                comment_pos = j
                break

        if comment_pos is not None:
            before = line[:comment_pos].rstrip()
            if before:
                result.append(before)
            # else: full-line comment — skip entirely
        else:
            result.append(line)

        pos += len(line) + 1  # +1 for the \n

    return '\n'.join(result)


def anonymize_code(source: str, fn_name: str | None = None) -> tuple[str, dict[str, str]]:
    """Anonymize user-defined identifiers in Python source code.

    Args:
        source: Python source code
        fn_name: The entry-point function name (will be renamed to "my_func").
                 If None, all user-defined functions are renamed to func_1, func_2, ...

    Returns:
        (anonymized_source, rename_map)

    Raises:
        SyntaxError: If source cannot be parsed
    """
    tree = ast.parse(source)
    collector = _IdentifierCollector(fn_name)
    collector.visit(tree)

    rename_map: dict[str, str] = {}

    if fn_name is not None:
        # Entry function → my_func
        rename_map[fn_name] = "my_func"

        # Other functions → helper_1, helper_2, ...
        for i, name in enumerate(collector.functions, 1):
            rename_map[name] = f"helper_{i}"
    else:
        # No entry function — rename all functions to func_1, func_2, ...
        for i, name in enumerate(collector.functions, 1):
            rename_map[name] = f"func_{i}"

    # Classes → MyClass, MyClass_2, ...
    for i, name in enumerate(collector.classes, 1):
        if i == 1:
            rename_map[name] = "MyClass"
        else:
            rename_map[name] = f"MyClass_{i}"

    # Parameters → arg_1, arg_2, ...
    for i, name in enumerate(collector.params, 1):
        rename_map[name] = f"arg_{i}"

    # Variables → var_1, var_2, ...
    for i, name in enumerate(collector.variables, 1):
        rename_map[name] = f"var_{i}"

    source_clean = _strip_comments(source)
    anonymized = _apply_renames(source_clean, rename_map)
    return anonymized, rename_map
