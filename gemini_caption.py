"""HI-VIS — background scene descriptions from Gemini.

The detector says what PPE is or isn't there; this says what the photo shows,
in plain prose, for the site record. It's a network round-trip and slower than
local inference, so every description is fired off the moment a batch lands and
collected later: submit() queues the work on a thread pool and returns
immediately, get() hands back the text once (and only once) it's ready.

State lives at module level, not in st.session_state, because worker threads
have no Streamlit script run context. Keys are image MD5s — the same photo
uploaded twice, in one session or two, costs one API call.

One POST per image to the Interactions API (/v1beta/interactions), which
replaced generateContent as the default interface in June 2026. No SDK:
requests is already a dependency.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import view_helpers as vh

# The API wants a fully-qualified "models/..." name; tolerate a bare one so
# HIVIS_GEMINI_MODEL=gemini-3.7-flash doesn't fail with an opaque 400.
MODEL = os.environ.get("HIVIS_GEMINI_MODEL", "models/gemini-3.7-flash")
if not MODEL.startswith("models/"):
    MODEL = f"models/{MODEL}"
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Thinking tokens are drawn from this same budget. Measured on
# gemini-3.7-flash at thinking_level "low": thoughts median 205, max 476;
# prose 20-94. At 500 that truncated 2 of 10 sample images — raise this if
# you see trailing "…" on descriptions.
MAX_OUTPUT_TOKENS = int(os.environ.get("HIVIS_GEMINI_MAX_TOKENS", "500"))
TOP_P = 0.95
THINKING_LEVEL = "low"

# Deliberately neutral: no safety or compliance judgement, so a reader never
# mistakes Gemini's prose for the detector's verdict. The two are different
# claims and only one of them is measured.
PROMPT = (
    "Describe this construction site photograph in one or two plain sentences "
    "for a site record: what the scene is, roughly how many people are visible "
    "and what they appear to be doing, and the setting. Describe only what is "
    "visible. Do not assess safety, compliance or risk. No preamble, no bullet "
    "points."
)

# ponytail: 4 workers is a guess tuned to a demo-sized batch, not a measurement.
# Raise it if large batches feel slow; Gemini's own rate limit is the real ceiling.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini-caption")
_futures = {}  # md5 -> Future[str]

# Observed in practice, not theory: a 4-wide batch reliably draws the odd 503
# "this model is currently experiencing high demand" that succeeds on a retry
# a second later. Without this, a normal batch shows scattered failures.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_ATTEMPTS = 3

# When the API refuses, it refuses every image the same way (bad key, no
# credit, model withdrawn). The per-tile caption only has room for a status
# code, so the server's own explanation is kept here and shown once, above the
# gallery — twelve identical cryptic tiles tell you nothing.
last_error = None


def api_key():
    """The GEMINI_API_KEY from .streamlit/secrets.toml, or None."""
    import streamlit as st

    try:
        return st.secrets.get("GEMINI_API_KEY") or None
    except Exception:
        return None


def key_problem():
    """Why there's no usable key, in words, or None when there is one.

    Worth its own function: st.secrets raises the same way for a missing file
    and an unparseable one, so an unquoted key in secrets.toml (TOML wants
    GEMINI_API_KEY = "...", quotes included) otherwise looks exactly like no
    key at all, and you go hunting in the wrong place."""
    import streamlit as st

    try:
        if st.secrets.get("GEMINI_API_KEY"):
            return None
    except Exception as exc:
        return f"`.streamlit/secrets.toml` could not be read — {type(exc).__name__}: {exc}"
    return "No `GEMINI_API_KEY` found in `.streamlit/secrets.toml`."


def submit(key, image):
    """Queue a description for one image (a PIL.Image). Call from the main
    thread — the key is read here and passed down, so workers never touch
    Streamlit state. Idempotent: safe to call for the whole batch on every
    rerun, which is what makes a key added mid-session take effect."""
    if key in _futures:
        return
    secret = api_key()
    if not secret:
        return
    _futures[key] = _pool.submit(_describe, image, secret)


def get(key):
    """The description for `key`, or None if it's still in flight or was never
    submitted. Failures come back as text, not exceptions — a broken API key
    should degrade one caption, not the page."""
    fut = _futures.get(key)
    if fut is None or not fut.done():
        return None
    try:
        return fut.result()
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return f"Description unavailable ({'HTTP %d' % code if code else type(exc).__name__})."


def pending(keys):
    """How many of `keys` are still being described — drives the polling
    fragment on the demo page, which stops rendering once this hits zero."""
    return sum(1 for k in keys if k in _futures and not _futures[k].done())


def _build_body(b64):
    return {
        "model": MODEL,
        "input": [
            {"type": "text", "text": PROMPT},
            {"type": "image", "mime_type": "image/jpeg", "data": b64},
        ],
        "generation_config": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "top_p": TOP_P,
            "thinking_level": THINKING_LEVEL,
        },
    }


def _parse(data):
    """Pull the prose out of an Interactions response.

    The reply is a list of `steps`; the model's own reasoning arrives as
    `thought` steps and must be skipped — only `model_output` is the answer.
    A response cut short by max_output_tokens comes back as
    status "incomplete" rather than "completed"."""
    text = "".join(
        part.get("text", "")
        for step in data.get("steps") or []
        if step.get("type") == "model_output"
        for part in step.get("content") or []
    ).strip()
    text = text.removeprefix("Description:").lstrip(": -*").strip()
    if not text:
        return "No description returned."
    # Say so rather than presenting half a sentence as the whole one.
    if data.get("status") != "completed":
        return text.rstrip(".,;: ") + "…"
    return text


def _describe(image, secret):
    global last_error
    import requests

    # Downscale on the worker thread, not the main one. 768px is plenty for a
    # two-sentence description and keeps the upload small. b64_image() copies
    # before touching the image, so sharing it across threads is safe.
    b64 = vh.b64_image(image, max_dim=768, quality=80)
    body = _build_body(b64)
    for attempt in range(_ATTEMPTS):
        last = attempt == _ATTEMPTS - 1
        try:
            response = requests.post(URL, headers={"x-goog-api-key": secret}, json=body, timeout=120)
        except requests.exceptions.RequestException:
            # Timeouts and dropped connections are the same kind of transient
            # as a 503, and were observed alongside them — retry, don't fail.
            if last:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code in _RETRY_STATUS and not last:
            time.sleep(2 ** attempt)  # 1s, 2s — transient overload clears fast
            continue
        if not response.ok:
            try:
                detail = response.json()["error"]["message"]
            except Exception:
                detail = response.reason
            last_error = f"HTTP {response.status_code} — {detail}"
        response.raise_for_status()  # anything else (401, 400, 404) is a real fault, not a wait
        last_error = None
        return _parse(response.json())


if __name__ == "__main__":
    body = _build_body("QUJD")
    assert body["model"].startswith("models/")
    assert body["input"][1] == {"type": "image", "mime_type": "image/jpeg", "data": "QUJD"}
    assert body["generation_config"] == {
        "max_output_tokens": MAX_OUTPUT_TOKENS, "top_p": 0.95, "thinking_level": "low"}

    def _steps(text, status="completed"):
        return {"status": status, "steps": [
            {"type": "thought", "signature": "abc"},  # must never leak into the caption
            {"type": "model_output", "content": [{"type": "text", "text": text}]}]}

    assert _parse(_steps(" Two workers. ")) == "Two workers."
    assert _parse(_steps(": Two workers.")) == "Two workers."
    assert _parse(_steps("Description: Two workers.")) == "Two workers."
    # A thought-only reply is not an answer.
    assert _parse({"status": "completed", "steps": [{"type": "thought", "signature": "x"}]}) \
        == "No description returned."
    assert _parse({"status": "completed", "steps": []}) == "No description returned."
    assert _parse({}) == "No description returned."
    # Truncated replies are flagged, not passed off as complete.
    assert _parse(_steps("Two workers on a scaff", status="incomplete")) == "Two workers on a scaff…"

    assert pending(["nope"]) == 0
    assert 503 in _RETRY_STATUS and 400 not in _RETRY_STATUS
    print("gemini_caption self-check OK")
