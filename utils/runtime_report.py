"""Sanitized runtime evidence without importing startup or loading any models."""
import os
import resource
import sys
import time
import threading
try:
    import psutil
except ImportError:
    psutil = None

_resource_lock = threading.Lock()
_last_swap = None


def memory_sample():
    """Best-effort counters; unavailable OS metrics remain null, not zero."""
    global _last_swap
    result = {"rss_bytes": None, "system_available_bytes": None,
              "system_used_percent": None, "swap_used_bytes": None, "swap_delta_bytes": None}
    if psutil is None:
        return result
    try:
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        result.update(rss_bytes=psutil.Process().memory_info().rss,
                      system_available_bytes=memory.available, system_used_percent=memory.percent,
                      swap_used_bytes=swap.used)
        with _resource_lock:
            if _last_swap is not None:
                result["swap_delta_bytes"] = swap.used - _last_swap
            _last_swap = swap.used
    except (OSError, psutil.Error):
        pass
    return result

_CONFIG_KEYS = (
    "LEAN_BRAIN_ENABLED", "LEAN_CONTEXT_STATE_ENABLED", "LEAN_IMPULSE_MENU_ENABLED",
    "LEAN_BRAIN_TRANSCRIPT_TURNS", "LEAN_BRAIN_TRANSCRIPT_TURNS_MAX",
    "LLM_MODEL", "LLM_CONVERSATION_MODEL", "TRANSCRIPTION_BACKEND",
    "MEMORY_SEMANTIC_RECALL_ENABLED", "MEMORY_SEMANTIC_EMBED_MODEL",
    "MEMORY_RETRIEVAL_BUDGET_SECS", "CONVERSATION_ARC_ENABLED",
    "CONVERSATION_ARC_BACKEND", "GAP_MERGE_ENABLED", "GAP_CATCHUP_ENABLED",
    "MOTION_HEADING_ALTERNATIVES_ENABLED", "NO_AUDIO_MODE",
    "CONTINUOUS_REPLY_CAPTURE_ENABLED", "SPEAKER_ID_SEGMENT_CHECK_ENABLED",
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
            "identity": "utterance evidence resolver; legacy candidate adapters",
            "input": ("concurrent reply capture; serial response owner; recovery at seams"
                      if getattr(config_module, "CONTINUOUS_REPLY_CAPTURE_ENABLED", False)
                      else "serial response owner; recovery capture"),
            "physical_safety": "Python guards and ESP32",
        },
        "loaded_models": loaded,
        "resources": {**memory_sample(), "peak_rss_bytes": peak_bytes,
                      "user_cpu_secs": usage.ru_utime, "system_cpu_secs": usage.ru_stime},
    }
