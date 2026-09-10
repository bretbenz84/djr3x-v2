"""Opt-in speedup: cache the reference prefix across calls.

A robot clones one fixed character, so every request re-encodes the identical
reference: the transcript through the text encoder, the clip through the audio
tokenizer, then the codebook-EOS marker. On the 19.5 s reference at 8-bit that
is ~240 ms of the time-to-first-audio, paid again on every single line.

`_prompt_embeddings` builds

    [ speaker+ref_text ] [ ref_audio codes ] [ EOS ] [ speaker+target_text ]
    \___________________ fixed per character ______/ \__ varies per call __/

so the prefix is cached keyed on (speaker, ref_text, reference identity) and
only the target-text segment is computed per call. Voice-direction prompts
(`instruct`) only ever change that trailing segment, so they share the cache.

Ported from Local/breeze-tts-2/scripts/ref_cache.py; production cache is bounded
and owned by each model instance.

Usage:
    from audio.breeze_ref_cache import enable_ref_cache
    enable_ref_cache()
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
from mlx_audio.tts.models.breeze_tts.breeze_tts import Model

_ORIGINAL = Model._prompt_embeddings
_PATCHED = False
_MAX_REFERENCES = 8
# Stored on each model, so unloaded engines and their prefixes can be collected.


def _reference_key(ref_audio) -> Optional[tuple]:
    """Identity for a reference clip: path plus mtime/size, or array contents."""
    if isinstance(ref_audio, (str, Path)):
        p = Path(ref_audio)
        try:
            st = p.stat()
            return ("path", str(p.resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            return ("path", str(p))
    if isinstance(ref_audio, mx.array):
        # Arrays are passed in rarely; hashing the bytes keeps this correct
        # rather than keying on an id() that can be recycled.
        return ("array", ref_audio.shape, bytes(memoryview(ref_audio.astype(mx.float32))).__hash__())
    return None


def _prompt_embeddings_cached(
    self,
    text: str,
    *,
    voice: Optional[str],
    instruct: Optional[str],
    ref_audio: Optional[Union[str, Path, mx.array]],
    ref_text: Optional[str],
) -> mx.array:
    if isinstance(ref_audio, (list, tuple)):
        if len(ref_audio) != 1:
            raise ValueError(
                "Breeze supports exactly one reference audio item per generation."
            )
        ref_audio = ref_audio[0]
    if isinstance(ref_text, (list, tuple)):
        if len(ref_text) != 1:
            raise ValueError(
                "Breeze supports exactly one reference transcript per generation."
            )
        ref_text = ref_text[0]
    if ref_audio is None:
        return _ORIGINAL(self, text, voice=voice, instruct=instruct,
                         ref_audio=ref_audio, ref_text=ref_text)
    if not ref_text:
        raise ValueError("Breeze voice cloning requires ref_text with ref_audio.")

    speaker = self._speaker(voice)
    ref_key = _reference_key(ref_audio)
    key = (speaker, ref_text, ref_key)
    cache = getattr(self, "_rex_reference_cache", None)
    if cache is None:
        cache = self._rex_reference_cache = OrderedDict()
    prefix = cache.get(key) if ref_key is not None else None
    if prefix is not None:
        cache.move_to_end(key)
    if prefix is None:
        ref_hidden = self.text_encoder(self._text_ids(f"{speaker}{ref_text}")[None, :])
        parts = [self.text_encoder_proj(ref_hidden),
                 self.backbone_model.embed_tokens(self._encode_reference(ref_audio))]
        eos_codes = mx.full((1, 1, self.num_codebooks),
                            self.config.codebook_eos_token_id)
        parts.append(self.backbone_model.embed_tokens(eos_codes))
        prefix = mx.concatenate(parts, axis=1)
        mx.eval(prefix)
        if ref_key is not None:
            cache[key] = prefix
            while len(cache) > _MAX_REFERENCES:
                cache.popitem(last=False)

    target = f"{speaker}{text}"
    if instruct:
        target = f"{speaker}<ins_bos>{instruct}<ins_eos>{text}"
    tail = self.text_encoder_proj(self.text_encoder(self._text_ids(target)[None, :]))
    return mx.concatenate([prefix, tail], axis=1)


def warm(model, ref_audio, ref_text: str, voice: str = "S0") -> None:
    """Populate the cache before the first request arrives."""
    _prompt_embeddings_cached(model, "warm", voice=voice, instruct=None,
                              ref_audio=ref_audio, ref_text=ref_text)


def enable_ref_cache() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    Model._prompt_embeddings = _prompt_embeddings_cached
    _PATCHED = True
    return True


def disable_ref_cache() -> bool:
    global _PATCHED
    if not _PATCHED:
        return False
    Model._prompt_embeddings = _ORIGINAL
    _PATCHED = False
    return True


def clear(model) -> None:
    cache = getattr(model, "_rex_reference_cache", None)
    if cache is not None:
        cache.clear()
