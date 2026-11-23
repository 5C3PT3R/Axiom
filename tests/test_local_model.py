"""Local-model plumbing that needs no GPU: tool-call parsing, chat conversion, rounding of tool output."""
import json

from axiom.agents.investigation import _round
from axiom.llm.local import openai_tools, parse, to_chat


def test_parse_tool_calls():
    text = ('thinking\n<tool_call>\n{"name": "drill_down", "arguments": {"dimension": "os"}}\n</tool_call>'
            '<tool_call>{not json}</tool_call>')
    content, calls = parse(text)
    assert content == "thinking" and calls == [{"name": "drill_down", "arguments": {"dimension": "os"}}]


def test_to_chat_round_trip():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "t", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]}]
    chat = to_chat("sys", msgs)
    assert [m["role"] for m in chat] == ["system", "user", "assistant", "tool"]
    assert chat[2]["tool_calls"][0]["function"]["arguments"] == {"a": 1}  # dict, as Qwen's template needs
    assert openai_tools([{"name": "t", "description": "d", "input_schema": {}}])[0]["function"]["name"] == "t"


def test_round_keeps_json_valid():
    out = _round({"a": float("nan"), "b": 32160.575, "c": 0.992083, "count": 5, "s": [1.23456]})
    assert out == {"a": None, "b": 32161, "c": 0.9921, "s": [1.235]}
    json.loads(json.dumps(out, allow_nan=False))
