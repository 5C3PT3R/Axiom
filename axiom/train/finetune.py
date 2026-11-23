"""LoRA fine-tune of a small instruct model on the teacher's conversations (fits a 6 GB laptop GPU).

Loss is taken only on assistant turns, and the LM head is applied only at those positions: the full
(seq x 151k-vocab) logits tensor for a 6k-token conversation would not fit in 6 GB.
"""
import json
import math
import random
import time
from pathlib import Path

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def encode(tok, rec: dict, max_len: int):
    """(input_ids, labels) with labels = -100 outside assistant turns; None if too long."""
    msgs, tools = rec["messages"], rec["tools"]
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            a = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
            b = tok.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False)
            if not full.startswith(b) or not b.startswith(a):
                return None  # template not prefix-stable for this record; skip rather than mislabel
            spans.append((len(a), len(b)))
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [t if any(s <= o[0] < e for s, e in spans) else -100 for t, o in zip(ids, enc["offset_mapping"])]
    return ids, labels


def train(data: Path = Path("data/train/train.jsonl"), out: Path = Path("models/axiom-lora"),
          base: str = BASE, epochs: int = 2, lr: float = 2e-4, accum: int = 8, max_len: int = 6144,
          rank: int = 16, seed: int = 0) -> dict:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "needs a CUDA GPU (pip install torch --index-url .../whl/cu126)"
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    recs = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in (encode(tok, x, max_len) for x in recs) if r]
    print(f"{len(rows)}/{len(recs)} conversations fit in {max_len} tokens", flush=True)
    random.Random(seed).shuffle(rows)
    n_val = max(10, len(rows) // 20)
    val, tr = rows[:n_val], rows[n_val:]

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    inner, head = model.base_model.model.model, model.base_model.model.lm_head

    def loss_of(ids, labels):
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor(labels[1:], device="cuda")
        h = inner(input_ids=x).last_hidden_state[0, :-1]
        keep = y != -100
        return F.cross_entropy(head(h[keep]).float(), y[keep])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = math.ceil(len(tr) * epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (
        1 + math.cos(math.pi * min(s, steps) / steps)))

    def evaluate():
        model.eval()
        with torch.no_grad():
            v = sum(loss_of(*r).item() for r in val) / len(val)
        model.train()
        return v

    log, t0, step = [], time.time(), 0
    print(f"train {len(tr)}  val {len(val)}  steps {steps}  val_loss {evaluate():.4f}", flush=True)
    model.train()
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(tr)
        run = 0.0
        for i, r in enumerate(tr, 1):
            loss = loss_of(*r) / accum
            loss.backward()
            run += loss.item()
            if i % accum == 0 or i == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"epoch {ep + 1} step {step}/{steps} loss {run / 10:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                    run = 0.0
        vl = evaluate()
        log.append({"epoch": ep + 1, "val_loss": vl, "seconds": round(time.time() - t0)})
        print(log[-1], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    info = {"base": base, "train": len(tr), "val": len(val), "epochs": epochs, "lr": lr, "rank": rank,
            "log": log}
    (out / "training.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
