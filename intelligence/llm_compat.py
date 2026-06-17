"""
intelligence/llm_compat.py — OpenAI model-compatibility shim.

ONE choke point for translating chat-completion parameters per model family, so the
conversation can migrate from `gpt-4o-mini` to a GPT-5-class model (e.g. `gpt-5.4-mini`)
by changing config instead of editing dozens of call sites. It is **behavior-neutral
for gpt-4o-mini** (passes parameters through unchanged), so wiring a call site through
it today changes nothing until a GPT-5 model is selected.

Why a shim is needed — GPT-5 / o-series *reasoning* models change the API contract:
  - `max_tokens` is rejected; the parameter is renamed `max_completion_tokens`.
  - A non-default `temperature` is rejected (HTTP 400) on reasoning models. Newer ones
    MAY accept it when `reasoning_effort` is "none"/omitted, but that is unconfirmed for
    gpt-5.4-mini — so by default we DROP `temperature` for GPT-5 models, gated on the
    `LLM_GPT5_PASS_TEMPERATURE` config flag for testing.
  - `reasoning_effort` (none|minimal|low|medium|high|xhigh) and `verbosity`
    (low|medium|high) are GPT-5-only knobs; injected from args or config when present.

This module is intentionally PURE + tiny: `prepare_chat_params()` does the translation
and is fully unit-testable with no network (see tests/test_llm_compat.py); `create()`
is a thin convenience that applies it and calls the given client.

Full plan, hybrid-rollout strategy, and A/B method: docs/gpt-5_4_mini.md.
"""

from __future__ import annotations

from typing import Any, Optional

import config


# Model-name prefixes whose API contract differs from gpt-4o (reasoning models).
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: Optional[str]) -> bool:
    """True for GPT-5 / o-series reasoning models that change the parameter contract
    (max_completion_tokens instead of max_tokens, restricted temperature, reasoning
    knobs). False for gpt-4o-mini and other classic chat models."""
    m = (model or "").strip().lower()
    return any(m.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


def conversation_model() -> str:
    """The model for Rex's user-facing in-character generation. Defaults to LLM_MODEL,
    so the conversation can be flipped to a GPT-5-class model independently of the
    classifier/router/vision calls (hybrid rollout)."""
    return getattr(config, "LLM_CONVERSATION_MODEL", None) or config.LLM_MODEL


def prepare_chat_params(
    *,
    model: str,
    messages: list,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[dict] = None,
    stream: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    timeout: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Return the kwargs dict for `client.chat.completions.create(...)` for `model`,
    translating the GPT-5-family parameter differences. PURE — no network, no client.

    For a non-reasoning model (gpt-4o-mini) the output is exactly the classic param set
    (`max_tokens`, `temperature` passed through), so routing a call site through this is
    a no-op until a GPT-5 model is selected.
    """
    params: dict[str, Any] = {"model": model, "messages": messages}
    # Pass-through params that are identical across families.
    if response_format is not None:
        params["response_format"] = response_format
    if stream is not None:
        params["stream"] = stream
    if timeout is not None:
        params["timeout"] = timeout
    if extra:
        params.update(extra)

    if is_reasoning_model(model):
        # 1) Token cap: max_tokens -> max_completion_tokens.
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens
        # 2) GPT-5-only knobs: explicit arg wins, else config default, else omit.
        effort = reasoning_effort if reasoning_effort is not None else getattr(
            config, "LLM_REASONING_EFFORT", None
        )
        if effort:
            params["reasoning_effort"] = effort
        verb_value = verbosity if verbosity is not None else getattr(config, "LLM_VERBOSITY", None)
        if verb_value:
            params["verbosity"] = verb_value
        # 3) Temperature: dropped by default (reasoning models reject non-default temp);
        #    only forwarded once LLM_GPT5_PASS_TEMPERATURE is confirmed-safe and enabled.
        if temperature is not None and bool(getattr(config, "LLM_GPT5_PASS_TEMPERATURE", False)):
            params["temperature"] = temperature
    else:
        # Classic chat models (gpt-4o-mini): unchanged contract.
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
    return params


def create(client, **kwargs):
    """Prepare params for `kwargs['model']` and call the OpenAI client. Returns the raw
    response (or stream when `stream=True`). `client` is any OpenAI client instance, so
    every module keeps using its own configured client/timeouts/retries."""
    params = prepare_chat_params(**kwargs)
    return client.chat.completions.create(**params)
