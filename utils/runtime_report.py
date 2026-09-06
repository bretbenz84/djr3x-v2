"""Sanitized runtime evidence without importing startup or loading any models."""
import os
import resource
import sys
import time

_CONFIG_KEYS = (
    "LEAN_BRAIN_ENABLED", "LEAN_CONTEXT_STATE_ENABLED", "LEAN_IMPULSE_MENU_ENABLED",
    "LEAN_BRAIN_TRANSCRIPT_TURNS", "LEAN_BRAIN_TRANSCRIPT_TURNS_MAX",
    "LLM_MODEL", "LLM_CONVERSATION_MODEL", "TRANSCRIPTION_BACKEND",
    "MEMORY_SEMANTIC_RECALL_ENABLED", "MEMORY_SEMANTIC_EMBED_MODEL",
    "MEMORY_RETRIEVAL_BUDGET_SECS", "CONVERSATION_ARC_ENABLED",
    "CONVERSATION_ARC_BACKEND", "GAP_MERGE_ENABLED", "GAP_CATCHUP_ENABLED",
    "MOTION_HEADING_ALTERNATIVES_ENABLED", "NO_AUDIO_MODE",
)


def snapshot(config_module=None):
    if config_module is None:
        import config as config_module
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Darwin reports bytes, Linux reports KiB. This is peak RSS, not current
    # RSS or system pressure; do not label it as either of those measurements.
    peak_bytes = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    loaded = {}
    for module_name, fields in {
        "audio.transcription": ("_qwen_model",),
        "audio.speaker_id": ("_encoder",),
        "audio.vad": ("_model",),
        "audio.local_tts": ("_model",),
    }.items():
        module = sys.modules.get(module_name)
        loaded[module_name] = {
            "imported": module is not None,
            "model_present": any(getattr(module, field, None) is not None for field in fields),
        }
    return {
        "pid": os.getpid(), "at": time.time(),
        "effective_config": {key: getattr(config_module, key, None) for key in _CONFIG_KEYS},
        "owners": {
            "reply": "lean" if getattr(config_module, "LEAN_BRAIN_ENABLED", False) else "classic",
            "identity": "legacy ladder with shadow uncertainty gates",
            "input": "serial capture with bounded catch-up queue",
            "physical_safety": "Python guards and ESP32",
        },
        "loaded_models": loaded,
        "resources": {"peak_rss_bytes": peak_bytes,
                      "user_cpu_secs": usage.ru_utime, "system_cpu_secs": usage.ru_stime},
    }
