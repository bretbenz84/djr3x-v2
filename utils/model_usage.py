"""Request-owned usage reporting. Missing provider usage is unknown, never zero.

Stream contract: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
No prompts, credentials, prices or guessed token totals are logged here.
"""
import logging
import time
import uuid
from utils import turn_trace


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class RequestUsage:
    def __init__(self, model, purpose):
        self.owner = turn_trace.current()
        self.model, self.purpose = model, purpose
        self.request_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.usage = None
        self.done = False
        self.count("requests")

    def count(self, name, n=1):
        turn_trace.count_for(self.owner, f"usage.{self.purpose}.{name}", n)

    def observe(self, response):
        usage = _get(response, "usage")
        if usage is None:
            return
        values = {k: _get(usage, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if not all(isinstance(v, int) and v >= 0 for v in values.values()):
            return
        cached = _get(_get(usage, "prompt_tokens_details"), "cached_tokens")
        if isinstance(cached, int) and cached >= 0:
            values["cached_tokens"] = cached
        self.usage = values

    def finish(self, outcome):
        if self.done:
            return
        self.done = True
        self.count(outcome)
        if self.usage is None:
            self.count("usage_unknown")
        else:
            for key, value in self.usage.items():
                self.count(key, value)
        logging.getLogger(__name__).info(
            "[model_usage] request=%s turn=%s model=%s purpose=%s outcome=%s elapsed_ms=%d usage=%s",
            self.request_id, getattr(self.owner, "turn_id", None), self.model, self.purpose,
            outcome, round((time.monotonic()-self.started)*1000), self.usage)


class UsageStream:
    def __init__(self, source, request):
        self.source, self.request = source, request
        self.iterator = iter(source)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            chunk = next(self.iterator)
        except StopIteration:
            self.request.finish("completed")
            self.close()
            raise
        except BaseException:
            self.request.finish("failed")
            self.close()
            raise
        self.request.observe(chunk)
        return chunk

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()
        finally:
            self.request.finish("cancelled")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
