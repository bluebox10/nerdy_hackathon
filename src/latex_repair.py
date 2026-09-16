"""Undo JSON's interpretation of LaTeX commands as control characters.

json.loads reads "\\frac" written with a single backslash as formfeed + "rac".
The generator now repairs escapes before parsing, but rows produced before that fix
still carry the control characters. In maths text the mapping back is unambiguous
enough to reconstruct by looking at the letters that follow.
"""
import re

# control char -> (following letters, LaTeX command)
_RECOVER = [
    ("\x0c", "rac", "\\frac"), ("\x0c", "orall", "\\forall"),
    ("\x08", "egin", "\\begin"), ("\x08", "ar", "\\bar"),
    ("\t", "imes", "\\times"), ("\t", "ext", "\\text"), ("\t", "heta", "\\theta"),
    ("\t", "an", "\\tan"), ("\t", "riangle", "\\triangle"),
    ("\n", "eq", "\\neq"), ("\n", "ot", "\\not"),
    ("\r", "ight", "\\right"), ("\r", "ightarrow", "\\rightarrow"),
]
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f]")


def repair_latex(s):
    if not isinstance(s, str):
        return s
    for ch, tail, cmd in _RECOVER:
        # cmd already ends with `tail` (\x0c+"rac" -> "\\frac"), so the tail is consumed
        s = s.replace(ch + tail, cmd)
    s = s.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    s = _CTRL.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
