"""Shared evidence for the deterministic clock fast path."""
import re

TIME_QUERY_RE = re.compile(
    r"\bwhat\s+time(?:\s+is\s+it)?(?:\s+(?:now|again|please|right\s+now))*[?!.]*\s*$|"
    r"\b(?:what(?:'s| is)|tell\s+me|give\s+me|do\s+you\s+know)\s+"
    r"(?:the\s+)?(?:current\s+|exact\s+)?time(?:\s+(?:now|again|please))*[?!.]*\s*$",
    re.IGNORECASE,
)
