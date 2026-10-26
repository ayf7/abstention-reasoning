"""Restore the superscripts LeetCodeDataset lost when it flattened statements to text.

(It also left a literal backslash-r at line ends in 39 statements; those are dropped.)

The dataset writes 10^5 as "105", 2^31 - 1 as "231 - 1", 10^-5 as "10-5" and
n^2 as "n2". There is no intact copy to recover them from, so they are restored
with rules, written in the caret form the dataset already uses where it is
intact ("10^5", "2^31 - 1", "10^9 + 7", "10^-5", "O(n^2)", "2^n"):

  - Constraints section: every 10^3..10^18 and 2^31/2^32/2^63/2^64 token. A
    survey of all 2,701 problems found no genuine literal of these forms there.
  - Everywhere: "109 + 7", "O(n2)", "within/than 10-5", "231 - 1", "[-231,".
  - Everything else (prose powers, "2n", "n2", examples): a per-problem table,
    checked by hand. Look-alikes that are genuine values ("231" as a
    permutation, "2n people", "10-13 digits") are left alone.
"""
import re

_POW10 = r"10(?:1[0-8]|[3-9])"
_POW2 = r"2(?:31|32|63|64)"
_CONSTRAINT_POW = re.compile(rf"(?<![\w.^])({_POW10}|{_POW2})(?![\w^])")
_MOD = re.compile(r"(?<![\w.^])109[\s\xa0]*\+[\s\xa0]*7\b")
_BIG_O = re.compile(r"\bO\(([a-z])2\)")
_TOLERANCE = re.compile(r"\b(within|than) 10-(\d)\b")
_INT_MAX = re.compile(rf"(?<![\w.^])({_POW2})(?=\s*[-−]\s*1\b)")
_LITERAL_CR = re.compile(r"\\r(?=\n|$)")
_INT_MIN = re.compile(rf"\[([-−])({_POW2})(?=,)")

# question_id -> [(old, new)]; each `old` must occur, and every occurrence is replaced.
_BY_QUESTION = {
    8: [("than -231", "than -2^31"), ("to -231", "to -2^31")],
    29: [("than -231", "than -2^31"), ("return -231", "return -2^31")],
    59: [("1 to n2", "1 to n^2")],
    62: [("2 * 109", "2 * 10^9")],
    63: [("2 * 109", "2 * 10^9")],
    89: [("of 2n integers", "of 2^n integers"), ("[0, 2n - 1]", "[0, 2^n - 1]")],
    166: [("less than 104", "less than 10^4")],
    222: [("1 and 2h nodes", "1 and 2^h nodes")],
    231: [("n == 2x", "n == 2^x")],
    326: [("n == 3x", "n == 3^x"), ("3x = ", "3^x = ")],
    342: [("n == 4x", "n == 4^x")],
    378: [("k <= n2", "k <= n^2")],
    394: [("exceed 105", "exceed 10^5")],
    537: [("i2", "i^2")],
    544: [("n == 2x", "n == 2^x")],
    778: [("<\xa0n2", "<\xa0n^2")],
    779: [("2n - 1", "2^n - 1")],
    842: [("< 231,", "< 2^31,")],
    850: [("is 1018 modulo", "is 10^18 modulo")],
    909: [("n2", "n^2")],
    958: [("1 and 2h nodes", "1 and 2^h nodes")],
    1545: [("2n - 1", "2^n - 1")],
    1780: [("y == 3x", "y == 3^x")],
    1870: [("exceed 107", "exceed 10^7")],
    1925: [("a2 + b2 = c2", "a^2 + b^2 = c^2")],
    1982: [("all 2n subset", "all 2^n subset"), ("== 2n", "== 2^n")],
    2003: [("There are 105 genetic", "There are 10^5 genetic"), ("range [1, 105]", "range [1, 10^5]")],
    2052: [(")2", ")^2")],
    2197: [("equal to 108", "equal to 10^8")],
    2201: [("min(n2,", "min(n^2,")],
    2258: [("return 109.", "return 10^9."), ("Thus, 109 is", "Thus, 10^9 is")],
    2509: [("[1, 2n - 1 - 1]", "[1, 2^(n - 1) - 1]"), ("2n - 1", "2^n - 1")],
    2525: [("equal to 104", "equal to 10^4"), ("equal to 109", "equal to 10^9"),
           ("<= 109.", "<= 10^9."), ("8 * 106", "8 * 10^6")],
    2549: [("109 days", "10^9 days")],
    2571: [("x == 2i", "x == 2^i")],
    2588: [("Subtract 2k", "Subtract 2^k")],
    2594: [("r * n2", "r * n^2")],
    2647: [("into n2 unit", "into n^2 unit")],
    2749: [("subtract 2i + num2", "subtract 2^i + num2")],
    2939: [("< 2n.", "< 2^n.")],
    2967: [("less than 109.", "less than 10^9.")],
    3020: [("[x, x2, x4, ..., xk/2, xk, xk/2, ..., x4, x2, x]",
            "[x, x^2, x^4, ..., x^(k/2), x^k, x^(k/2), ..., x^4, x^2, x]")],
}


def _pow(m: re.Match) -> str:
    t = m.group(1)
    return f"10^{t[2:]}" if t.startswith("10") else f"2^{t[1:]}"


def repair_statement(question_id: int, text: str) -> str:
    text = _LITERAL_CR.sub("", text)
    head, sep, tail = text.rpartition("Constraints:")
    if sep:
        text = head + sep + _CONSTRAINT_POW.sub(_pow, tail)
    text = _MOD.sub("10^9 + 7", text)
    text = _BIG_O.sub(r"O(\1^2)", text)
    text = _TOLERANCE.sub(r"\1 10^-\2", text)
    text = _INT_MAX.sub(lambda m: f"2^{m.group(1)[1:]}", text)
    text = _INT_MIN.sub(lambda m: f"[{m.group(1)}2^{m.group(2)[1:]}", text)
    for old, new in _BY_QUESTION.get(question_id, ()):
        if old not in text:
            raise ValueError(f"question {question_id}: {old!r} not in statement")
        text = text.replace(old, new)
    return text
