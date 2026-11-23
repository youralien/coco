import json
from types import SimpleNamespace

import pytest
from external_api.litellm_api import FunctionCall, LiteLLMMessage, ToolCall
from fastapi import FastAPI
from fastapi.testclient import TestClient
from proactive_tutor import pi_bridge


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("COCO_PI_BRIDGE_TOKEN", "secret")
    calls = []
    agent = SimpleNamespace(
        model="configured/model",
        prompt="Be helpful.",
        _tool_definitions=lambda: [
            {"type": "function", "function": {"name": "observe_screen"}}
        ],
        _execute_tool_call=lambda call: (
            calls.append(call) or {"observation": "A spreadsheet"}
        ),
    )
    tutor = SimpleNamespace(
        tutor_agent=agent,
        problem_statement="Current task",
        memory="User memory",
        conversation_history=["Earlier suggestion"],
        _chat_messages=[],
        _user_name_context_block=lambda: "User",
        _ai_tools_context_block=lambda: "Tools",
        _curriculum_context_block=lambda: "Curriculum",
    )
    app = FastAPI()
    app.include_router(pi_bridge.create_pi_router(lambda: tutor))
    return TestClient(app), tutor, calls


def test_bridge_requires_desktop_token(bridge):
    client, _, _ = bridge
    assert client.get("/agent/context").status_code == 401
    assert (
        client.get(
            "/agent/context", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    response = client.get("/agent/context", headers={"Authorization": "Bearer secret"})
    assert response.json()["tools"][0]["function"]["name"] == "observe_screen"


def test_bridge_uses_existing_tools_and_records_turn_for_other_coco_flows(bridge):
    client, tutor, calls = bridge
    client.headers["Authorization"] = "Bearer secret"
    response = client.post(
        "/agent/tools/execute",
        json={"name": "observe_screen", "arguments": {"focus": "error"}},
    )
    assert response.json()["observation"] == "A spreadsheet"
    assert calls == [{"name": "observe_screen", "arguments": {"focus": "error"}}]
    response = client.post("/agent/turn", json={"user_text": "Do it", "text": "Done"})
    assert response.json()["history_length"] == 3
    assert tutor._chat_messages[-1] == {"role": "assistant", "content": "Done"}


def test_completion_preserves_tools_and_streaming_while_using_configured_model(
    bridge, monkeypatch
):
    client, _, _ = bridge
    client.headers["Authorization"] = "Bearer secret"
    received = {}

    def completion(messages, **kwargs):
        received.update(kwargs)
        kwargs["on_chunk"]("Working…")
        return LiteLLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=FunctionCall(
                        name="write", arguments='{"path":"out.txt","content":"ok"}'
                    ),
                )
            ],
        ), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    monkeypatch.setattr(pi_bridge, "chat_completion", completion)
    tools = [
        {
            "type": "function",
            "function": {"name": "write", "parameters": {"type": "object"}},
        }
    ]
    response = client.post(
        "/agent/v1/chat/completions",
        json={
            "model": "untrusted/override",
            "messages": [{"role": "user", "content": "Do it"}],
            "tools": tools,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert received["model"] == "configured/model"
    assert received["tools"] == tools
    assert any(
        event["choices"][0]["delta"].get("content") == "Working…" for event in events
    )
    assert (
        events[-2]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        == "write"
    )
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["total_tokens"] == 15
    assert response.text.endswith("data: [DONE]\n\n")


def test_completion_failure_is_reported_in_stream(bridge, monkeypatch):
    client, _, _ = bridge
    client.headers["Authorization"] = "Bearer secret"

    def fail(*args, **kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(pi_bridge, "chat_completion", fail)
    response = client.post("/agent/v1/chat/completions", json={"messages": []})
    assert "Provider unavailable" in response.text
    assert '"error"' in response.text


def test_execution_sessions_isolate_model_task_history_and_turns(bridge):
    client, foreground, _ = bridge
    client.headers["Authorization"] = "Bearer secret"
    for session_id in ("a", "b"):
        result = client.post(
            f"/agent/sessions/{session_id}",
            json={
                "model": f"model-{session_id}",
                "problem": f"task-{session_id}",
                "messages": [{"role": "user", "text": f"history-{session_id}"}],
            },
        )
        assert result.status_code == 200
        assert result.json()["model"] == f"model-{session_id}"
        assert f"task-{session_id}" in result.json()["profile"]
    client.post(
        "/agent/turn",
        headers={"X-Coco-Session": "a"},
        json={"user_text": "only A", "text": "A finished"},
    )
    a = client.get("/agent/context", headers={"X-Coco-Session": "a"}).json()
    b = client.get("/agent/context", headers={"X-Coco-Session": "b"}).json()
    assert any("A finished" in message for message in a["history"])
    assert not any("A finished" in message for message in b["history"])
    assert not any("history-a" in message for message in b["history"])
    assert foreground.conversation_history == ["Earlier suggestion"]
    assert (
        client.get("/agent/context", headers={"X-Coco-Session": "unknown"}).status_code
        == 404
    )


def test_only_foreground_execution_updates_coaching_history(bridge):
    client, foreground, _ = bridge
    client.headers["Authorization"] = "Bearer secret"
    for session_id in ("a", "b"):
        client.post(f"/agent/sessions/{session_id}", json={"model": "fake"})
    client.post("/agent/active/b")
    for session_id in ("a", "b"):
        client.post(
            "/agent/turn",
            headers={"X-Coco-Session": session_id},
            json={"user_text": session_id, "text": f"reply-{session_id}"},
        )
    assert any("reply-b" in message for message in foreground.conversation_history)
    assert not any("reply-a" in message for message in foreground.conversation_history)


@pytest.mark.parametrize("enabled", [False, True])
def test_qwen_thinking_options_and_response_separation(bridge, monkeypatch, enabled):
    client, tutor, _ = bridge
    tutor.tutor_agent.model = "hosted_vllm/nvidia/Qwen3.8-27B-NVFP4"
    client.headers["Authorization"] = "Bearer secret"
    received = {}

    def completion(messages, **kwargs):
        received.update(kwargs)
        kwargs["on_reasoning_chunk"]("Private reasoning.")
        kwargs["on_chunk"]("Final answer.")
        return LiteLLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call-1", function=FunctionCall(name="read", arguments="{}")
                )
            ],
        ), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    monkeypatch.setattr(pi_bridge, "chat_completion", completion)
    request = {"messages": [{"role": "user", "content": "Hi"}]}
    if enabled:
        request["chat_template_kwargs"] = {"enable_thinking": True}
    response = client.post("/agent/v1/chat/completions", json=request)
    assert received["extra_body"]["chat_template_kwargs"]["enable_thinking"] is enabled
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [event["choices"][0]["delta"] for event in events]
    assert "".join(d.get("content", "") for d in deltas) == "Final answer."
    assert (
        "".join(d.get("reasoning_content", "") for d in deltas) == "Private reasoning."
    )
    assert any(d.get("tool_calls") for d in deltas)
    assert events[-1]["usage"]["total_tokens"] == 15


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one Helvetica text line per page."""
    count = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>",
    ]
    font = 3 + 2 * count
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {4 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return out


def test_pdf_extract_returns_page_text_from_raw_bytes(bridge):
    client, _, _ = bridge
    pdf = make_pdf(["Quarterly revenue grew", "Second page notes", ""])
    assert client.post("/agent/pdf/extract", content=pdf).status_code == 401
    client.headers["Authorization"] = "Bearer secret"

    full = client.post("/agent/pdf/extract", content=pdf).json()
    assert full["total_pages"] == 3
    assert [p["number"] for p in full["pages"]] == [1, 2, 3]
    assert full["pages"][0]["text"] == "Quarterly revenue grew"
    assert "No extractable text" in full["pages"][2]["text"]

    window = client.post(
        "/agent/pdf/extract?first_page=2&last_page=2", content=pdf
    ).json()
    assert window["first_page"] == window["last_page"] == 2
    assert window["pages"] == [{"number": 2, "text": "Second page notes"}]


def test_pdf_extract_rejects_invalid_input(bridge):
    client, _, _ = bridge
    client.headers["Authorization"] = "Bearer secret"
    garbage = client.post("/agent/pdf/extract", content=b"not a pdf at all")
    assert garbage.status_code == 422
    out_of_range = client.post(
        "/agent/pdf/extract?first_page=5", content=make_pdf(["Only page"])
    )
    assert out_of_range.status_code == 422
    assert "out of range" in out_of_range.json()["detail"]
