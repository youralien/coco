"""Cheap checks that keep a broken observer model from polluting sensing.

Two failure modes are guarded against:

* **Degenerate output.** Small or mis-served models can fall into repetition
  loops ("EmailDraftEmailDraft...") and run to the token cap. Such output must
  not reach observation history, where it would be fed back into the next
  prompt and make the next observation worse.
* **Dropped images.** An OpenAI-compatible server can accept image parts and
  silently discard them (e.g. a chat-template/model-type mismatch), leaving a
  "vision" model to hallucinate a screen it never saw. A differential probe
  detects this: the same prompt with and without an image must differ in
  prompt tokens.
"""

from __future__ import annotations

import base64
import struct
import zlib

from external_api.llm import chat_completion

# zlib compression ratio (raw bytes / compressed bytes). Healthy observer JSON
# measured 1.65-2.25 across 519 stored observations; the degenerate repetition
# loops measured 4.7-27. 3.0 leaves a wide margin on both sides.
MAX_COMPRESSION_RATIO = 3.0
# Short strings compress poorly and give a meaningless ratio.
_MIN_RATIO_BYTES = 256


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / len(zlib.compress(raw))


def degenerate_output_reason(
    text: str, *, output_tokens: int, max_tokens: int
) -> str | None:
    """Return why ``text`` is unusable as an observation, or ``None`` if fine."""
    if max_tokens > 0 and output_tokens >= max_tokens:
        return f"hit the output token cap ({output_tokens}/{max_tokens} tokens)"
    if len(text.encode("utf-8")) >= _MIN_RATIO_BYTES:
        ratio = compression_ratio(text)
        if ratio > MAX_COMPRESSION_RATIO:
            return (
                f"repetitive output (compression ratio {ratio:.1f} > "
                f"{MAX_COMPRESSION_RATIO})"
            )
    return None


def _solid_png(size: int = 64) -> bytes:
    """A small opaque grey PNG, built without an imaging dependency."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    row = b"\x00" + b"\x80\x80\x80" * size  # filter byte + RGB pixels
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


_PROBE_IMAGE_URL = "data:image/png;base64," + base64.b64encode(_solid_png()).decode()
_PROBE_TEXT = "Reply with one word."


def _probe_prompt_tokens(model: str, *, with_image: bool) -> int:
    content: list[dict] = [{"type": "text", "text": _PROBE_TEXT}]
    if with_image:
        content.append({"type": "image_url", "image_url": {"url": _PROBE_IMAGE_URL}})
    _, metrics = chat_completion(
        messages=[{"role": "user", "content": content}],
        model=model,
        max_tokens=1,
        operation="observer_image_probe",
    )
    return int(metrics.get("prompt_tokens", 0) or 0)


def probe_image_input(model: str) -> bool | None:
    """Return whether ``model``'s endpoint feeds images to the model.

    ``True``/``False`` when the probe is conclusive; ``None`` when it could not
    run (endpoint down, provider reports no usage), in which case callers
    should proceed as before rather than block sensing. Blocking: two tiny
    completions with ``max_tokens=1``.
    """
    try:
        text_only = _probe_prompt_tokens(model, with_image=False)
        with_image = _probe_prompt_tokens(model, with_image=True)
    except Exception:
        return None
    if text_only <= 0 or with_image <= 0:
        return None
    return with_image > text_only
