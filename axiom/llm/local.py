"""Local fine-tuned model (Qwen + LoRA adapter) behind the same `messages.create` shape as the API.

Training (axiom.train) and inference share `to_chat`/`openai_tools`, so the model sees at run time exactly
the format it was trained on. Tool calls use Qwen's native `<tool_call>{json}</tool_call>` format.
"""
import json
import re

from axiom.llm.openai_compat import _to_anthropic, _to_openai_messages

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def openai_tools(specs: list) -> list:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in specs]


def to_chat(system: str, messages: list) -> list:
    """Anthropic-shaped conversation -> chat-template messages (tool arguments as dicts, as Qwen expects)."""
    out = _to_openai_messages(system, messages)
    for m in out:
        for c in m.get("tool_calls") or []:
            c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    return out


def parse(text: str) -> tuple[str, list[dict]]:
    """Generated text -> (plain text, [{name, arguments}]). Malformed calls are dropped."""
    calls = []
    for blob in _CALL.findall(text):
        try:
            c = json.loads(blob)
            calls.append({"name": c["name"], "arguments": c.get("arguments") or {}})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return _CALL.sub("", text).strip(), calls


class _Messages:
    def __init__(self, adapter: str, base: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(adapter)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
        self.model = PeftModel.from_pretrained(model, adapter).merge_and_unload().to(dev).eval()
        self.dev, self.torch = dev, torch

    def create(self, model, system, messages, tools, temperature=0, max_tokens=4096, tool_choice=None):
        text = self.tok.apply_chat_template(to_chat(system, messages), tools=openai_tools(tools),
                                            add_generation_prompt=True, tokenize=False)
        forced = ""
        if tool_choice and tool_choice.get("name"):  # force the call by starting it for the model
            forced = f'<tool_call>\n{{"name": "{tool_choice["name"]}", "arguments": '
        ids = self.tok(text + forced, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            gen = self.model.generate(**ids, max_new_tokens=min(max_tokens, 2048), do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new = gen[0, ids["input_ids"].shape[1]:]
        out = forced + self.tok.decode(new, skip_special_tokens=True)
        content, calls = parse(out)
        return _to_anthropic({
            "choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": {
                "content": content or None,
                "tool_calls": [{"id": f"call_{i}", "function": {"name": c["name"],
                                                                "arguments": json.dumps(c["arguments"])}}
                               for i, c in enumerate(calls)]}}],
            "usage": {"prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": len(new)}})


class LocalAPI:
    def __init__(self, adapter: str, base: str):
        self.messages = _Messages(adapter, base)
