#!/usr/bin/env python3
"""
tools/gpt5_smoke_test.py — LIVE OpenAI smoke test for a GPT-5-class model.

Hits the REAL OpenAI API once per call-SHAPE to find out what actually works BEFORE
flipping Rex's conversation model. This is the real gate: the unittest suite mocks the
LLM, so it stays green and tells you NOTHING about real API breakage. This does.

It is intentionally NOT part of the unittest suite — it needs network + an API key and
costs a few cents. Run it by hand:

    venv/bin/python tools/gpt5_smoke_test.py
    venv/bin/python tools/gpt5_smoke_test.py --model gpt-5.4-mini --effort none --pass-temp
    venv/bin/python tools/gpt5_smoke_test.py --vision

The headline question it settles: does <model> accept `temperature` (with
`reasoning_effort=none`), or does it 400? Run once WITHOUT --pass-temp (shim drops
temperature) and once WITH it; whichever passes tells you what to set
`LLM_GPT5_PASS_TEMPERATURE` to. Full plan: docs/gpt-5_4_mini.md.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import apikeys  # noqa: E402
import config  # noqa: E402
from intelligence import llm_compat  # noqa: E402


def _tiny_png_data_url() -> str:
    """A 16x16 solid-blue PNG as a data URL, so the vision shape needs no local file."""
    try:
        from PIL import Image  # Pillow ships with the project
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (40, 90, 200)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        # 1x1 transparent PNG fallback.
        b = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        return "data:image/png;base64," + base64.b64encode(b).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-5.4-mini", help="model id to smoke-test")
    ap.add_argument("--effort", default="none", help="reasoning_effort: none|minimal|low|medium|high")
    ap.add_argument("--pass-temp", action="store_true", help="forward temperature to the model")
    ap.add_argument("--vision", action="store_true", help="also run the vision (image_url) shape")
    args = ap.parse_args()

    # Override the shim config for THIS run only (process-local).
    config.LLM_REASONING_EFFORT = args.effort or None
    config.LLM_GPT5_PASS_TEMPERATURE = bool(args.pass_temp)

    from openai import OpenAI
    client = OpenAI(api_key=apikeys.OPENAI_API_KEY, timeout=30.0, max_retries=1)
    model = args.model

    print(f"\nSmoke-testing model={model!r}  reasoning_effort={args.effort!r}  "
          f"pass_temp={args.pass_temp}\n" + "-" * 64)

    results: list[tuple[str, bool]] = []

    def run(name: str, fn) -> None:
        t0 = time.monotonic()
        try:
            detail = fn()
            dt = time.monotonic() - t0
            print(f"  PASS  {name:<22} {dt:5.1f}s  {detail}")
            results.append((name, True))
        except Exception as exc:  # noqa: BLE001 — we want to report every failure shape
            dt = time.monotonic() - t0
            print(f"  FAIL  {name:<22} {dt:5.1f}s  {type(exc).__name__}: {exc}")
            results.append((name, False))

    # 1) Plain chat — does the basic call + max_completion_tokens rename work?
    def plain():
        r = llm_compat.create(
            client, model=model,
            messages=[{"role": "user", "content": "Say hi in three words."}],
            max_tokens=20,
        )
        return repr((r.choices[0].message.content or "").strip())
    run("plain chat", plain)

    # 2) Streaming — the main conversation path consumes chunk.choices[0].delta.
    def streaming():
        out = []
        stream = llm_compat.create(
            client, model=model,
            messages=[{"role": "user", "content": "Count to three."}],
            max_tokens=20, stream=True,
        )
        for ch in stream:
            delta = ch.choices[0].delta
            if getattr(delta, "content", None):
                out.append(delta.content)
        return repr("".join(out).strip())
    run("streaming", streaming)

    # 3) temperature=0 — THE critical question for the deterministic classifier/router
    #    calls. With --pass-temp the shim forwards temperature; without it, the shim
    #    drops it (so this just confirms the call still works at default temp).
    def temp_zero():
        r = llm_compat.create(
            client, model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10, temperature=0.0,
        )
        txt = repr((r.choices[0].message.content or "").strip())
        note = "temperature FORWARDED" if config.LLM_GPT5_PASS_TEMPERATURE else "temp dropped by shim (use --pass-temp to test forwarding)"
        return f"{txt}  [{note}]"
    run("temperature=0", temp_zero)

    # 4) JSON structured output — used by sentiment/intent/memory-extraction calls.
    def json_mode():
        r = llm_compat.create(
            client, model=model,
            messages=[{"role": "user", "content": 'Return only this JSON: {"ok": true}'}],
            max_tokens=30, response_format={"type": "json_object"},
        )
        return repr(json.loads(r.choices[0].message.content or "{}"))
    run("response_format json", json_mode)

    # 5) Vision (optional) — only relevant if you later migrate VISION_MODEL too.
    if args.vision:
        def vision():
            r = llm_compat.create(
                client, model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "What color is this image? One word."},
                    {"type": "image_url", "image_url": {"url": _tiny_png_data_url(), "detail": "low"}},
                ]}],
                max_tokens=10,
            )
            return repr((r.choices[0].message.content or "").strip())
        run("vision image_url", vision)

    passed = sum(1 for _, ok in results if ok)
    print("-" * 64)
    print(f"{passed}/{len(results)} shapes passed.\n")
    print("Interpreting it:")
    print("  - If 'temperature=0' PASSED with --pass-temp, this model accepts temperature →")
    print("    you can keep your deterministic temps; set LLM_GPT5_PASS_TEMPERATURE=True.")
    print("  - If it FAILED (e.g. 400 unsupported) with --pass-temp, leave the flag False")
    print("    (the shim drops temperature for GPT-5 models).")
    print("  - Watch the per-shape seconds: reasoning models add latency. If 'streaming' is")
    print("    slow, try --effort minimal/none for the conversation path.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
