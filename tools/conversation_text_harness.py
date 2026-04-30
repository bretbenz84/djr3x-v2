#!/usr/bin/env python3
"""
Run conversational logic from plain text turns, without microphone, Whisper, or TTS.

This is a diagnostic harness, not the robot runtime. It feeds typed user turns
through the local command parser, intent classifier, agenda builder, social
frame, and LLM response path, then records what Rex would have said. It can
also run a closed-loop actor that replies naturally to Rex for several turns.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import apikeys  # noqa: E402
import config  # noqa: E402
from intelligence import (  # noqa: E402
    command_parser,
    conversation_agenda,
    intent_classifier,
    social_frame,
)
from intelligence import interaction  # noqa: E402
from memory import conversations as conv_memory  # noqa: E402
from memory import people as people_memory  # noqa: E402
from openai import OpenAI  # noqa: E402
from world_state import world_state  # noqa: E402

_actor_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


def _load_turns(args: argparse.Namespace) -> list[str]:
    turns: list[str] = []
    if args.file:
        path = Path(args.file)
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
            if not isinstance(data, list):
                raise SystemExit("JSON scenario file must be a list of strings or objects.")
            for item in data:
                if isinstance(item, str):
                    turns.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    turns.append(str(item["text"]))
                else:
                    raise SystemExit(f"Unsupported scenario item: {item!r}")
        else:
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    turns.append(stripped)
    turns.extend(args.turn or [])
    if not turns and args.mode != "actor":
        raise SystemExit("Provide turns as arguments or with --file.")
    return turns


def _resolve_person(args: argparse.Namespace) -> tuple[int | None, str | None]:
    if args.person_id is not None:
        row = people_memory.get_person(args.person_id)
        return args.person_id, (row or {}).get("name") or args.person
    if not args.person:
        return None, None

    row = people_memory.find_person_by_name(args.person)
    if row:
        return int(row["id"]), str(row.get("name") or args.person)
    if args.create_person:
        person_id, _created = people_memory.find_or_create_person(args.person)
        return person_id, args.person
    raise SystemExit(
        f"No person named {args.person!r} found. Use --create-person to make a test row."
    )


def _prime_world_state(person_id: int | None, name: str | None) -> None:
    if person_id is None:
        world_state.update("people", [])
        world_state.update("crowd", {
            "count": 0,
            "count_label": "alone",
            "dominant_speaker": None,
            "last_updated": None,
        })
        return

    world_state.update("people", [{
        "id": f"person_{person_id}",
        "person_db_id": person_id,
        "face_id": name,
        "voice_id": name,
        "position": "center",
    }])
    world_state.update("crowd", {
        "count": 1,
        "count_label": "one_on_one",
        "dominant_speaker": name,
        "last_updated": None,
    })
    try:
        interaction.consciousness.mark_engagement(person_id)
    except Exception:
        pass


def _frame_preview(
    text: str,
    person_id: int | None,
    answered_question: dict | None = None,
) -> tuple[str, social_frame.SocialFrame]:
    agenda = conversation_agenda.build_turn_directive(
        text,
        person_id,
        answered_question=answered_question,
    )
    frame = social_frame.build_frame(
        text,
        person_id,
        answered_question=answered_question,
        agenda_directive=agenda,
    )
    return agenda, frame


def _dummy_filler_timer() -> threading.Event:
    event = threading.Event()
    return event


def _run_turn(
    text: str,
    *,
    person_id: int | None,
    person_name: str | None,
    no_llm: bool,
) -> dict[str, Any]:
    spoken: list[dict[str, Any]] = []

    def fake_speak(spoken_text: str, *args: Any, **kwargs: Any) -> bool:
        spoken.append({
            "text": spoken_text,
            "emotion": kwargs.get("emotion") if "emotion" in kwargs else (
                args[0] if args else "neutral"
            ),
            "priority": kwargs.get("priority", 1),
        })
        return True

    conv_memory.add_to_transcript(person_name or "user", text)

    command = command_parser.parse(text)
    intent = "general"
    agenda = ""
    frame: social_frame.SocialFrame | None = None
    route = "llm"
    response = ""

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(interaction, "_speak_blocking", fake_speak))
        stack.enter_context(mock.patch.object(interaction, "_start_latency_filler_timer", _dummy_filler_timer))
        stack.enter_context(mock.patch.object(interaction.llm, "classify_surprise", return_value=False))
        if no_llm:
            stack.enter_context(mock.patch.object(
                interaction.llm,
                "get_response",
                side_effect=lambda prompt, *a, **k: f"[NO_LLM] {str(prompt).strip()[:160]}",
            ))

        if command is not None:
            route = f"command:{command.command_key}"
            response = interaction._execute_command(command, person_id, person_name, text)
        else:
            intent = intent_classifier.classify(text)
            if intent != "general":
                route = f"intent:{intent}"
                response = interaction._handle_classified_intent(intent, text, person_id) or ""
            else:
                agenda, frame = _frame_preview(text, person_id)
                response = interaction._stream_llm_response(text, person_id)

    if not response and spoken:
        response = spoken[-1]["text"]
    if response:
        conv_memory.add_to_transcript("Rex", response)

    if frame is None and route.startswith("llm"):
        agenda, frame = _frame_preview(text, person_id)

    return {
        "user": text,
        "route": route,
        "command": (
            {
                "key": command.command_key,
                "match_type": command.match_type,
                "args": command.args,
            }
            if command is not None
            else None
        ),
        "intent": intent,
        "frame": asdict(frame) if frame is not None else None,
        "agenda": agenda,
        "response": response,
        "spoken": spoken,
    }


def _write_outputs(results: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = out_dir / f"conversation_text_harness_{stamp}.jsonl"
    md_path = out_dir / f"conversation_text_harness_{stamp}.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = ["# Conversation Text Harness", ""]
    for idx, item in enumerate(results, 1):
        lines.extend([
            f"## Turn {idx}",
            "",
            f"**User:** {item['user']}",
            "",
            f"**Route:** `{item['route']}`",
            f"**Intent:** `{item['intent']}`",
        ])
        frame = item.get("frame") or {}
        if frame:
            lines.append(
                "**Frame:** "
                f"purpose=`{frame.get('purpose')}`, "
                f"questions=`{frame.get('allow_question')}`, "
                f"roast=`{frame.get('allow_roast')}`, "
                f"visual=`{frame.get('allow_visual_comment')}`"
            )
        lines.extend([
            "",
            f"**Rex:** {item.get('response') or '[no response]'}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return jsonl_path, md_path


def _generate_actor_reply(
    *,
    actor_name: str,
    persona: str,
    goal: str,
    conversation: list[dict[str, str]],
    rex_text: str,
    model: str,
) -> str:
    transcript = "\n".join(
        f"{turn['speaker']}: {turn['text']}" for turn in conversation[-10:]
    )
    system = (
        f"You are simulating {actor_name}, a human talking out loud to DJ-R3X. "
        "Reply as the human only, not as Rex. Keep replies natural for spoken "
        "conversation: usually 3-18 words, occasionally one sentence longer. "
        "Do not narrate actions. Do not quote Rex. If Rex asks a question, answer it. "
        "If Rex is pushy, repetitive, wrong, or cuts off oddly, react naturally. "
        "You may introduce ordinary topic shifts the way a real person would."
    )
    user = (
        f"Human persona:\n{persona}\n\n"
        f"Conversation goal / test focus:\n{goal}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        f"Rex just said:\n{rex_text}\n\n"
        "What does the human say next?"
    )
    resp = _actor_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        max_tokens=80,
    )
    return _clean_actor_reply(resp.choices[0].message.content or "", actor_name)


def _clean_actor_reply(text: str, actor_name: str) -> str:
    cleaned = (text or "").strip().strip('"').strip()
    if not cleaned:
        return ""
    name = re.escape((actor_name or "").strip())
    if name:
        cleaned = re.sub(rf"^\s*{name}\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(human|user|person|speaker)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^\s*[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3}\s*:\s*",
        "",
        cleaned,
    )
    return cleaned.strip().strip('"').strip()


def _load_seed(args: argparse.Namespace) -> list[str]:
    seeds = _load_turns(args)
    if seeds:
        return seeds
    return [args.seed]


def _run_actor_loop(
    args: argparse.Namespace,
    *,
    person_id: int | None,
    person_name: str | None,
) -> list[dict[str, Any]]:
    seeds = _load_seed(args)
    results: list[dict[str, Any]] = []
    conversation: list[dict[str, str]] = []
    user_text = seeds[0]
    actor_model = args.actor_model or config.LLM_MODEL

    for idx in range(1, args.turns + 1):
        print(f"[{idx}/{args.turns}] USER: {user_text}", flush=True)
        result = _run_turn(
            user_text,
            person_id=person_id,
            person_name=person_name,
            no_llm=args.no_llm,
        )
        rex_text = result.get("response") or ""
        print(f"      {result['route']} -> {rex_text or '[no response]'}", flush=True)
        results.append(result)
        conversation.append({"speaker": person_name or args.person or "Human", "text": user_text})
        if rex_text:
            conversation.append({"speaker": "Rex", "text": rex_text})

        if idx >= args.turns:
            break

        if idx < len(seeds):
            user_text = seeds[idx]
        else:
            user_text = _generate_actor_reply(
                actor_name=person_name or args.person or "the human",
                persona=args.actor_persona,
                goal=args.actor_goal,
                conversation=conversation,
                rex_text=rex_text,
                model=actor_model,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feed plain text into Rex conversation logic without speech/TTS.",
    )
    parser.add_argument(
        "--mode",
        choices=("batch", "actor"),
        default="batch",
        help="batch = provided turns only; actor = LLM human replies to Rex.",
    )
    parser.add_argument("turn", nargs="*", help="One or more user turns.")
    parser.add_argument("--file", help="Plain text file (one turn per line) or JSON list.")
    parser.add_argument("--person", default="Bret Benziger", help="Person name for context.")
    parser.add_argument("--person-id", type=int, help="Existing person_id to use.")
    parser.add_argument(
        "--create-person",
        action="store_true",
        help="Create --person if it does not already exist.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Stub LLM responses; useful for fast route/frame checks.",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/conversation_harness",
        help="Directory for JSONL and Markdown output.",
    )
    parser.add_argument(
        "--keep-transcript",
        action="store_true",
        help="Do not clear the in-memory transcript before starting.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=8,
        help="Actor mode: number of user turns to simulate.",
    )
    parser.add_argument(
        "--seed",
        default="Hey Rex, let's just talk for a bit.",
        help="Actor mode: first user utterance when no file/turn is supplied.",
    )
    parser.add_argument(
        "--actor-persona",
        default=(
            "A friendly but direct tester. They like natural banter, dislike "
            "being interrogated, and will correct Rex when it gets weird."
        ),
        help="Actor mode: human persona for generated replies.",
    )
    parser.add_argument(
        "--actor-goal",
        default=(
            "Have a natural short conversation and expose routing mistakes, "
            "cut-off questions, unwanted topic persistence, and false music intents."
        ),
        help="Actor mode: test focus for generated replies.",
    )
    parser.add_argument(
        "--actor-model",
        default=None,
        help="Actor mode: OpenAI model for simulated human replies.",
    )
    args = parser.parse_args()

    person_id, person_name = _resolve_person(args)
    _prime_world_state(person_id, person_name)
    if not args.keep_transcript:
        conv_memory.clear_transcript()

    if args.mode == "actor":
        results = _run_actor_loop(args, person_id=person_id, person_name=person_name)
    else:
        turns = _load_turns(args)
        results = []
        for idx, text in enumerate(turns, 1):
            print(f"[{idx}/{len(turns)}] USER: {text}", flush=True)
            result = _run_turn(
                text,
                person_id=person_id,
                person_name=person_name,
                no_llm=args.no_llm,
            )
            print(f"      {result['route']} -> {result.get('response') or '[no response]'}", flush=True)
            results.append(result)

    jsonl_path, md_path = _write_outputs(results, Path(args.out_dir))
    print(f"\nWrote:\n  {jsonl_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
