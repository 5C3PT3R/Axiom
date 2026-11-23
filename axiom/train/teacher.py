"""Expert trajectories for fine-tuning. A teacher that knows the injected cause plays the Investigation
agent's tool loop (looking at the true dimensions plus decoys, so the student learns to read the tool
outputs) and the Critic's verdict, producing exactly the conversations the local model will see.
"""
import json
import random
from datetime import date
from pathlib import Path

from axiom.agents import critic, investigation
from axiom.agents.critic import _run_all_code_checks
from axiom.agents.investigation import SUBMIT, TOOL_SPECS, _check, _context, _view
from axiom.analysis.drilldown import dims_for
from axiom.analysis.tools import SESSION_DIMS, Tools
from axiom.config import load_settings
from axiom.detection import detect
from axiom.evidence import EvidenceStore
from axiom.llm.local import openai_tools, to_chat
from axiom.models import Hypothesis
from axiom.quality import check_day


class _Episode:
    def __init__(self, tools: Tools, first_user: str):
        self.tools, self.messages, self.n = tools, [{"role": "user", "content": first_user}], 0

    def call(self, name: str, **args) -> dict:
        self.n += 1
        out = getattr(self.tools, name)(**args)
        tid = f"call_{self.n}"
        self.messages += [{"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name,
                                                             "input": args}]},
                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                                                        "content": _view(out)}]}]
        return out

    def finish(self, name: str, args: dict) -> None:
        self.messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"call_{self.n + 1}", "name": name, "input": args}]})


def _row(out: dict, seg: str) -> dict | None:
    return next((r for r in out["result"].get("segments", []) if str(r["segment"]) == seg), None)


def _statement(seg: str, row: dict, traffic: dict, share: float, mix: bool) -> str:
    tr = traffic["result"]
    sess = f"sessions in the segment were {tr['sessions']:,.0f} vs a baseline of {tr['baseline']:,.0f}"
    move = "fell" if row["value"] < row["baseline"] else "rose"
    if mix:
        why = f"{sess}, a traffic-mix shift toward this segment rather than a change in its behaviour"
    elif tr["within_normal"]:
        why = f"{sess}, so traffic was normal and the loss is in conversion or order value"
    else:
        why = f"{sess}, so the move is driven by traffic"
    return (f"Revenue in {seg} {move} from {row['baseline']:,.0f} to {row['value']:,.0f}; {why}. "
            f"This segment accounts for {abs(share) * 100:.1f}% of the revenue change.")


def investigation_episode(data_dir: Path, gt: dict, r: random.Random):
    """(messages, hypotheses, decoy hypothesis filters) for one incident, or None if not investigable."""
    day = date.fromisoformat(gt["date"])
    dq = check_day(data_dir, day)
    if dq.status == "BLOCK":
        return None
    store = EvidenceStore()
    tools = Tools(data_dir, store)
    anomalies = detect(data_dir, day, store, min_rel_change=load_settings()["detection"]["min_rel_change"])
    if not anomalies:
        return None
    dq_text = dq.status + "".join(f"; {i.check}: {i.detail}" for i in dq.issues)
    ep, d = _Episode(tools, _context(day, anomalies, "", dq_text)), str(day)
    truths = gt["expected_filters"]
    dec = ep.call("decompose_metric", metric="revenue", date=d)

    true_dims = list(dict.fromkeys(k for t in truths for k in t))
    decoys = r.sample([x for x in dims_for("revenue") if x not in true_dims], 2)
    first = true_dims + decoys
    r.shuffle(first)
    drills = {dim: ep.call("drill_down", metric="revenue", dimension=dim, date=d) for dim in first}

    hyps, chain0 = [], [a.evidence_id for a in anomalies[:1]] + [dec["evidence_id"]]
    for t in truths:
        # lead with the dimension whose true value carries the larger share, then drill the rest inside it
        dims = sorted(t, key=lambda k: -abs((_row(drills[k], t[k]) or {"share": 0})["share"]))
        used, f, last = [drills[dims[0]]], {dims[0]: t[dims[0]]}, drills[dims[0]]
        for k in dims[1:]:
            last = ep.call("drill_down", metric="revenue", dimension=k, date=d, filters=dict(f))
            used.append(last)
            f[k] = t[k]
        row = _row(last, t[dims[-1]])
        if row is None:
            return None
        sess_f = {k: v for k, v in t.items() if k in SESSION_DIMS}
        traffic = ep.call("check_traffic", date=d, filters=sess_f or None)
        if "payment_method" in t:
            ep.call("get_events", event_type="payment_failure", date=d, filters=sess_f or None,
                    group_by="payment_method")
        contrib = ep.call("segment_contribution", filters=dict(t), date=d)
        seg = ", ".join(f"{k}={v}" for k, v in t.items())
        ids = [last["evidence_id"], traffic["evidence_id"], contrib["evidence_id"]]
        hyps.append({"statement": _statement(seg, row, traffic, contrib["result"]["share"],
                                             gt.get("type") == "mix_shift"),
                     "segment_filters": dict(t), "evidence_ids": ids,
                     "chain": chain0 + [u["evidence_id"] for u in used] + [traffic["evidence_id"],
                                                                         contrib["evidence_id"]]})

    # the strongest decoy segment is named and dismissed with its own (small) share
    best = max(((dim, s) for dim in decoys for s in drills[dim]["result"]["segments"]),
               key=lambda x: abs(x[1]["share"]))
    rejected = [{"statement": f"{best[0]}={best[1]['segment']} explains only "
                              f"{abs(best[1]['share']) * 100:.1f}% of the change",
                 "evidence_ids": [drills[best[0]]["evidence_id"]]}]
    if any(_check(h, tools) for h in hyps + rejected):
        return None  # the teacher must itself pass the numeric verifier
    ep.finish("submit_hypotheses", {"hypotheses": hyps, "rejected_alternatives": rejected})
    return ep, hyps, {best[0]: str(best[1]["segment"])}, tools, dq


def critic_episode(tools: Tools, day: date, hyps: list[dict], decoy: dict, dq) -> list:
    """Critic conversation: accept the true hypotheses, reject the decoy."""
    items = [(h["segment_filters"], h["statement"], h["evidence_ids"], True) for h in hyps]
    c = tools.segment_contribution(decoy, str(day))
    seg = ", ".join(f"{k}={v}" for k, v in decoy.items())
    items.append((decoy, f"The revenue change is driven by {seg}.", [c["evidence_id"]], False))
    random.Random(str(day)).shuffle(items)
    objs = []
    for i, (f, st, ids, _) in enumerate(items, 1):
        share = c["result"]["share"] if f is decoy else tools.segment_contribution(f, str(day))["result"]["share"]
        objs.append(Hypothesis(id=f"H{i}", statement=st, segment_filters=f, chain=ids, evidence_ids=ids,
                               contribution_pct=100 * min(max(share, 0.0), 1.0)))
    dq_status = dq.status
    dq_issues = [i.model_dump() for i in dq.issues]
    checks = {h.id: _run_all_code_checks(tools, h, day, "weekday", dq_status, dq_issues) for h in objs}
    verdicts = []
    for h, (_, _, _, good) in zip(objs, items):
        failed = [x["detail"] for x in checks[h.id] if not x["passed"]]
        if good:
            reasons = [f"explains {h.contribution_pct:.1f}% of the revenue change"] + (
                [f"noted: {failed[0]}"] if failed else ["all checks passed"])
            verdicts.append({"hypothesis_id": h.id, "verdict": "accept", "reasons": reasons})
        else:
            top = max((o for o, it in zip(objs, items) if it[3]), key=lambda o: o.contribution_pct)
            verdicts.append({"hypothesis_id": h.id, "verdict": "reject", "reasons": [
                f"explains {h.contribution_pct:.1f}% of the change vs {top.contribution_pct:.1f}% for "
                f"{top.id}; it overlaps the real cause rather than causing the move"
                if h.contribution_pct >= 30 else
                f"explains only {h.contribution_pct:.1f}% of the change; the move sits in another segment"]})
    msgs = [{"role": "user", "content": critic._context(day, objs, checks, [])},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "submit_verdicts",
                                               "input": {"verdicts": verdicts}}]}]
    return msgs


def build(seeds: list[int], out: Path) -> dict:
    """Write out/train.jsonl: one record per conversation {kind, messages (chat format), tools}."""
    from axiom.train.incidents import make_dataset

    out.mkdir(parents=True, exist_ok=True)
    inv_tools, crit_tools = openai_tools(TOOL_SPECS + [SUBMIT]), openai_tools([critic.VERDICT_SPEC])
    stats = {"incidents": 0, "investigation": 0, "critic": 0}
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for seed in seeds:
            data_dir = make_dataset(seed)
            r = random.Random(seed)
            for gt in json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8")):
                stats["incidents"] += 1
                res = investigation_episode(data_dir, gt, r)
                if res is None:
                    continue
                ep, hyps, decoy, tools, dq = res
                fh.write(json.dumps({"kind": "investigation", "tools": inv_tools,
                                     "messages": to_chat(investigation.PROMPT, ep.messages)}) + "\n")
                cm = critic_episode(tools, date.fromisoformat(gt["date"]), hyps, decoy, dq)
                fh.write(json.dumps({"kind": "critic", "tools": crit_tools,
                                     "messages": to_chat(critic.PROMPT, cm)}) + "\n")
                stats["investigation"] += 1
                stats["critic"] += 1
            print(seed, stats, flush=True)
    return stats

