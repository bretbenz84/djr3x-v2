"""Eleven v3 audio-tag text helpers — the canonical tag pattern + strip.

Audio tags ([excited], [sighs], …) shape DELIVERY at synthesis time and may now
appear inline (mid-reply / mid-sentence) in text headed for the speech path:
authored on canned seam lines (repair_moves recovery tags) or emitted by the
lean brain. They must reach ElevenLabs only — never the transcript, GUI, logs,
or memory. This module is dependency-free so both audio.tts (synthesis side)
and utils.conv_log (display side) share ONE pattern instead of drifting copies.
"""

import re
from typing import Optional

# A bracketed tag word: letters plus the space/'/- v3 uses in multi-word tags
# ("[happy gasp]"). Deliberately narrow so stage brackets like "[unintelligible]"
# in HEARD text or numeric brackets are still matched/stripped conservatively.
AUDIO_TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z '\-]*)\]")


def strip_audio_tags(text: Optional[str]) -> str:
    """Remove [audio tags] from text — use anywhere Rex's line is stored or
    displayed, so a v3 delivery tag never leaks into the transcript/log/memory."""
    if not text:
        return text or ""
    return re.sub(r"\s{2,}", " ", AUDIO_TAG_RE.sub("", text)).strip()
