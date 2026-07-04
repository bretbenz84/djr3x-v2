"""
memory/fact_quality.py — content-quality gate for stored memories.

The extraction pipeline (intelligence/llm.py extract_facts/extract_interests/
extract_preferences) sometimes emits garbage VALUES: tautologies ('dad'->'dad'),
raw sentence fragments ('I might go see my dad...'), fiction plot scenes, negated
clauses stored as positive facts, and interests misattributed to Rex. add_fact()
and upsert_interest() are the two storage chokepoints; this module is the gate they
call just before INSERT.

DESIGN RULE: this is a live robot. A false positive ERASES a real memory. Every
predicate is deliberately narrow and conservative — when in doubt, KEEP. Semantic
classes a regex cannot see (a NAMED place the speaker never visited, a fabricated
pet name) are handled at the PROMPT layer; the gate is defense-in-depth for the
mechanical shapes and only claims the shapes it can actually see.

Root-caused + adversarially verified 2026-07-03 (workflow wf_0d93bffd): six defect
classes traced against the real garbage in Bret's record, each fix proven not to
drop the good facts (real hometown, real pet name, favorite movie, the specific
Beethoven movement, a stated dislike, a negated worldview).
"""

import re

# ── token helpers ────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _words(s: str) -> list[str]:
    return _TOKEN_RE.findall((s or "").lower())


def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    return " ".join((s or "").strip().lower().split()).strip(".,;:!?\"'")


# ── relation / collective nouns ──────────────────────────────────────────────
# When one of these is the WHOLE value, the fact restates the relation and carries
# no content: a family/pet fact's value must be a NAME or a concrete detail. The
# set is an explicit allowlist; a real name ('Rex', 'Max', 'Sacramento') is never
# in it, so real names always pass.
_RELATION_NOUNS = {
    "dad", "father", "papa", "pop", "stepdad", "stepfather",
    "mom", "mother", "mommy", "stepmom", "stepmother",
    "parent", "parents", "son", "daughter", "child", "children", "kid", "kids",
    "brother", "sister", "sibling", "siblings", "wife", "husband", "spouse",
    "partner", "boyfriend", "girlfriend", "fiance", "fiancee",
    "grandpa", "grandma", "grandfather", "grandmother",
    "aunt", "uncle", "cousin", "nephew", "niece",
    "friend", "friends", "roommate", "coworker", "colleague",
    "dog", "cat", "pet", "pets", "bird", "fish", "animal",
    # generic content-free stand-ins
    "thing", "stuff", "something", "someone", "people", "person", "family",
}


# Categories where a bare relation noun as the WHOLE value is meaningless garbage
# ('dad'->family, 'dog'->pet). EXCLUDES identity/appearance, where a relation-ish
# word is a legit enumerated value ('child'/'kid' as age_category, etc.).
_RELATION_TAUTOLOGY_CATEGORIES = {"family", "pet"}


def is_tautology(category: str, key: str, value: str) -> str | None:
    """REJECT when the value is a bare relation noun (in a family/pet fact) or
    literally echoes the key. dad='dad', dog='dog', family_member='dad'.
    Returns a reason or None. A NAME ('Rex'), any multi-word value, or a bare
    relation-ish word used as a real enumerated value ('child' as age_category)
    passes."""
    v = _norm(value)
    if not v:
        return "empty"
    k = _norm(key)
    if v == k:
        return "tautology_key"          # dog='dog', father='father' — garbage anywhere
    if _norm(category) in _RELATION_TAUTOLOGY_CATEGORIES and v in _RELATION_NOUNS:
        return "tautology_relation_noun"   # bare relation noun as a family/pet value
    return None
    # NOTE: we do NOT reject value==category — it kills zero real garbage and only
    # widens the false-positive surface (a legit one-word value that equals its
    # category label, e.g. a 'music' preference).


# ── first-person sentence fragments ──────────────────────────────────────────
# A structured fact VALUE should be a distilled noun phrase, not a whole clause the
# speaker uttered. We ANCHOR on a leading first-person subject so we only reject
# when the value STARTS as a spoken sentence — NOT when 'you'/'they' appears
# mid-string (that over-broad form wrongly ate 'I love The Beatles'-class values).
#
# Word-boundary fix vs. the original sketch: `('|\b)` after the pronoun so
# contracted/suffixed leads ('Its', 'Ive', 'Ill', 'Youre', 'Theyre') are caught.
# An optional leading filler/interjection ('oh I love to fix things', 'yeah I went
# camping') is skipped so a spoken clause with a discourse marker still reads as a
# fragment — the filler + a real first-person pronoun + >=4 tokens is never a
# distilled value.
_FIRST_PERSON_LEAD_RE = re.compile(
    r"^\s*(?:(?:oh|yeah|yep|well|so|um|uh|hmm|honestly|actually|basically|like|okay|ok)[,\s]+)?"
    r"(i|we|my|it|its|you|he|she|they)('|\b)",
    re.IGNORECASE,
)


def is_fragment(category: str, key: str, value: str) -> str | None:
    """REJECT a whole first-person clause dumped into a scalar fact slot
    ('I might go see my dad for the 4th of July', 'Its usually historical on me').
    Requires BOTH a first-person lead AND >=4 tokens, so short noun-phrase values
    ('Coney Island', 'classical music and soundtracks') never trip it, and a
    3rd-person distilled value ('Bret plans to visit his dad ...') passes because
    it leads with a proper noun, not a pronoun."""
    v = (value or "").strip()
    n = len(v.split())
    if _FIRST_PERSON_LEAD_RE.match(v) and n >= 4:
        return "sentence_fragment"
    return None


# ── negation / hypothetical / hearsay ────────────────────────────────────────
# These almost never survive to the VALUE (the extractor strips them and emits the
# bare noun — the Coney-Island failure). So this predicate takes the SOURCE
# UTTERANCE, not just the value, and is the gate's only reliable defense for that
# class. It is scoped by CATEGORY: only place/preference-style facts, NEVER
# belief/worldview ('I am not religious' -> worldview MUST survive).
_NEGATION_RE = re.compile(
    # 'no'/'not' intentionally EXCLUDED as bare words — they clip 'no-bake cookies'
    # and Title 'Never Let Me Go' is handled by the category scope + short-value gate.
    r"\b(never|n't|don't|dont|doesn't|didn't|wouldn't|won't|can't|cannot|"
    r"isn't|aren't|hardly|no longer|would never)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(
    r"\b(if|imagine|suppose|pretend|hypothetical|someday|wish|"
    r"i'd (?:love|like) to|planning to|thinking (?:of|about))\b",
    re.IGNORECASE,
)
_HEARSAY_RE = re.compile(
    r"\b(someone|somebody|they|people)\s+(told|said|says|mentioned|thinks?)\b"
    r"|\bi heard\b|\brumou?r\b",
    re.IGNORECASE,
)

# Categories where a negated/hypothetical SOURCE clause means the extracted value
# is spurious. Deliberately EXCLUDES belief/worldview/preference-dislike, where a
# negatively-phrased statement is the real fact.
_NEGATION_SCOPE_CATEGORIES = {"hometown", "job", "pet", "family", "other"}


def is_negated_or_hypothetical(category: str, key: str, value: str,
                               utterance: str = "", source: str = "") -> str | None:
    """REJECT when the SOURCE clause negates/hypothesizes/hearsays the extracted
    fact. Requires the source utterance (the value alone is polarity-stripped).
    Scoped to place/bio categories and to short values so it cannot clip a stated
    dislike ('I hate country music' -> preference) or a belief."""
    cat = _norm(category)
    if cat not in _NEGATION_SCOPE_CATEGORIES:
        return None
    text = utterance or ""
    if not text:
        return None   # no source clause available -> defer to the prompt layer
    # Only fire when the extracted VALUE is short (a bare extracted noun, the
    # failure shape) AND the source clause carries a negation/hypothetical marker.
    if len(value.split()) <= 6:
        if _NEGATION_RE.search(text):
            return "negation_source"
        if _HYPOTHETICAL_RE.search(text):
            return "hypothetical_source"
    if _HEARSAY_RE.search(text):
        return "hearsay_source"
    return None


# ── fiction / media plot scenes ──────────────────────────────────────────────
# A movie/show TITLE is a valid preference/interest; a SCENE or plot point is not.
# Runs on BOTH the fact path (fact #20 has category='interest') and interests.
# `plot` is anchored ('the plot' / 'plot of') so a real 'garden plot' interest is
# not swept up.
_FICTION_RE = re.compile(
    r"^\s*(the\s+)?(scene|part|bit|moment|episode|chapter|clip)\s+(where|when|in\s+which)\b"
    r"|\bcharacter\s+(who|that|named)\b"
    r"|\bthe plot\b|\bplot of\b|\bstoryline\b",
    re.IGNORECASE,
)


def is_fiction_scene(name_or_value: str) -> str | None:
    """REJECT a plot scene stored as an interest/bio fact
    ('scene where the kids figure out it's their dad...'). Bare titles
    ('Mrs. Doubtfire', 'Star Trek') do NOT match."""
    if _FICTION_RE.search(name_or_value or ""):
        return "fiction_scene"
    return None


# ── verbatim question shard (values and notes) ───────────────────────────────
# A whole question stored as a value/note ('are you gonna judge me for my pizza?').
# Narrowed per the verifier: only a value that ENDS as a question, not any string
# containing '?'.
_QUESTION_TAIL_RE = re.compile(r"\?\s*$")


def is_verbatim_question(text: str) -> str | None:
    """REJECT a value/note that ends as a question."""
    t = (text or "").strip()
    if t and _QUESTION_TAIL_RE.search(t):
        return "verbatim_question"
    return None


# ── Rex speaker-misattribution (interest NOTES only) ─────────────────────────
_MISATTRIB_RE = re.compile(
    r"\b(rex|dj[\- ]?r3x|r3x)\b[^.]*\b(mention|said|told|obsess|into|indicat)",
    re.IGNORECASE,
)


def is_rex_misattribution(notes: str) -> str | None:
    """REJECT an interest whose NOTE credits Rex as the source of the interest
    (interest#23 'music', notes='Rex mentioned being obsessed with music...')."""
    if _MISATTRIB_RE.search(notes or ""):
        return "rex_misattribution"
    return None


# ── top-level composed gates ─────────────────────────────────────────────────
# Interest-flavored fact categories where the fiction gate applies.
_INTEREST_FACT_CATEGORIES = {"interest", "interest_note"}


def reject_fact(category: str, key: str, value: str,
                utterance: str = "", source: str = "") -> str | None:
    """Central storage gate for add_fact(). Return a reason to REJECT, else None.
    Order: cheapest/safest first. Every branch is proven against GOOD facts."""
    if not _norm(value):
        return "empty"
    r = is_tautology(category, key, value)
    if r:
        return r
    if _norm(category) in _INTEREST_FACT_CATEGORIES:
        r = is_fiction_scene(value)
        if r:
            return r
    r = is_verbatim_question(value)
    if r:
        return r
    r = is_fragment(category, key, value)
    if r:
        return r
    r = is_negated_or_hypothetical(category, key, value, utterance, source)
    if r:
        return r
    return None


def reject_interest(name: str, notes: str = "") -> str | None:
    """Central storage gate for upsert_interest() NAME. Return reason to REJECT.
    (Note-cleaning is separate — see clean_interest_note — so a real interest with
    a junk note is KEPT, not dropped.)"""
    nm = (name or "").strip()
    r = is_fiction_scene(nm)
    if r:
        return r
    if len(_words(nm)) > 8:            # a plot line, not a hobby label; generous cap
        return "not_an_interest_name"
    r = is_rex_misattribution(notes)
    if r:
        return r
    return None


def clean_interest_note(notes: str) -> str:
    """Scrub a junk NOTE without dropping the interest. Returns '' to blank it."""
    if is_verbatim_question(notes):
        return ""
    if is_rex_misattribution(notes):
        return ""
    return (notes or "").strip()
