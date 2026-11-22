import json
import zlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sensing import observer_guard, segment_processor
from sensing.observer_guard import degenerate_output_reason, probe_image_input
from sensing.segment_processor import AiTutoringProcessor

HEALTHY = (
    "```json\n"
    + json.dumps(
        {
            "current_task": "Drafting a reply to a scheduling email in Gmail",
            "task_continuity": "Continuing the same reply as the previous snapshot",
            "evidence": "The compose window shows a half-written paragraph proposing "
            "Tuesday at 3pm, and the cursor is at the end of the second sentence.",
            "stuck": "not stuck",
            "support_opportunity": "none",
        },
        indent=2,
    )
    + "\n```"
)
LOOP = "The user is on Coco tabs email " + "EmailDraftEmail" * 400


def test_healthy_observation_passes():
    assert degenerate_output_reason(HEALTHY, output_tokens=210, max_tokens=4096) is None


def test_repetition_loop_is_rejected():
    reason = degenerate_output_reason(LOOP, output_tokens=900, max_tokens=4096)
    assert reason is not None and "repetitive" in reason


def test_hitting_token_cap_is_rejected():
    reason = degenerate_output_reason(HEALTHY, output_tokens=4096, max_tokens=4096)
    assert reason is not None and "token cap" in reason


def test_short_text_is_not_judged_by_ratio():
    assert (
        degenerate_output_reason("ok ok ok ok", output_tokens=4, max_tokens=4096)
        is None
    )


def test_probe_png_is_valid():
    png = observer_guard._solid_png(8)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    idat_len = int.from_bytes(png[33:37], "big")
    raw = zlib.decompress(png[41 : 41 + idat_len])
    assert len(raw) == 8 * (1 + 8 * 3)


def _fake_completion(text_tokens: int, image_tokens: int):
    def fake(*, messages, **_kwargs):
        has_image = any(p["type"] == "image_url" for p in messages[0]["content"])
        return None, {"prompt_tokens": image_tokens if has_image else text_tokens}

    return fake


def test_probe_detects_dropped_images(monkeypatch):
    monkeypatch.setattr(observer_guard, "chat_completion", _fake_completion(24, 24))
    assert probe_image_input("omlx/Inkling-2-Mini") is False


def test_probe_accepts_working_vision(monkeypatch):
    monkeypatch.setattr(observer_guard, "chat_completion", _fake_completion(24, 1912))
    assert probe_image_input("omlx/Qwen") is True


def test_probe_is_inconclusive_when_endpoint_fails(monkeypatch):
    def boom(**_kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(observer_guard, "chat_completion", boom)
    assert probe_image_input("omlx/Qwen") is None


def _processor(memory_engine=None, images=("shot.jpg",)):
    processor = AiTutoringProcessor(
        http_client=SimpleNamespace(),
        tutor_url="http://localhost:8081",
        ai_tutor_output_log="unused.log",
        observer_model="omlx/model",
        memory_engine=memory_engine,
    )
    processor._build_context_prompt = AsyncMock(return_value="context")
    processor._collect_images = lambda text: (text, list(images))
    processor._cleanup_consumed_screenshots = lambda paths: None
    return processor


@pytest.mark.asyncio
async def test_degenerate_observation_is_kept_out_of_history(monkeypatch):
    add_observation = AsyncMock(return_value=True)
    processor = _processor(SimpleNamespace(add_observation=add_observation))
    processor._image_input_ok["omlx/model"] = True
    monkeypatch.setattr(
        segment_processor,
        "_observe",
        lambda *_a, **_k: (LOOP, {"output_tokens": 4096}),
    )

    obs, _text, metrics = await processor._handle_observation(type="snapshot")

    assert obs == ""
    assert metrics["success"] is False and "token cap" in metrics["error"]
    assert not processor._observation_history
    add_observation.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_is_skipped_when_endpoint_drops_images(monkeypatch):
    processor = _processor()
    monkeypatch.setattr(segment_processor, "probe_image_input", lambda _m: False)

    def must_not_call(*_a, **_k):
        raise AssertionError("observer should not be called")

    monkeypatch.setattr(segment_processor, "_observe", must_not_call)

    obs, _text, metrics = await processor._handle_observation(type="snapshot")
    await processor._handle_observation(type="snapshot")

    assert obs == ""
    assert metrics["error"] == "the model endpoint drops image input"
    assert processor._image_input_ok == {"omlx/model": False}


@pytest.mark.asyncio
async def test_inconclusive_probe_still_observes(monkeypatch):
    processor = _processor()
    monkeypatch.setattr(segment_processor, "probe_image_input", lambda _m: None)
    monkeypatch.setattr(
        segment_processor,
        "_observe",
        lambda *_a, **_k: (HEALTHY, {"output_tokens": 210}),
    )

    obs, _text, _metrics = await processor._handle_observation(type="snapshot")

    assert obs == HEALTHY
    assert processor._observation_history[-1]["obs"] == HEALTHY


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("omlx/Inkling-2-Mini", True),
        ("hosted_vllm/thinkingmachines/Inkling-2-Mini", True),
        ("thinkingmachines/inkling-1", True),
        ("omlx/Qwen3.6-35B-A3B-oQ4e-mtp", False),
        ("hosted_vllm/Qwen/Qwen3-VL", False),
    ],
)
def test_is_inkling(model, expected):
    assert segment_processor._is_inkling(model) is expected
