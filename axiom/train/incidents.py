"""Random incidents for training data (the 28 hand-written scenarios stay held out for evaluation).

Each training dataset is a normal 180-day simulation with a random incident every 5 days from day 35.
Spacing of 5 keeps the same-weekday baseline (days -7..-28) free of other incidents.
"""
import random
from datetime import timedelta
from pathlib import Path

import yaml

from axiom.config import load_settings
from axiom.simulator.generator import simulate

VALUES = {
    "platform": ["web", "android", "ios"], "os": ["windows", "macos", "android", "ios"],
    "country": ["IN", "US", "BR", "UK", "DE"],
    "traffic_source": ["organic", "paid", "email", "direct", "social"],
    "device_type": ["desktop", "mobile"], "new_vs_returning": ["new", "returning"],
    "category": ["electronics", "fashion", "home", "beauty", "grocery"],
}
FUNNEL = ["atc_mult", "checkout_mult", "attempt_mult"]


def _segment(r: random.Random, dims: list[str]) -> dict:
    d = r.choice(dims)
    return {d: r.choice(VALUES[d])}


def _effect(r: random.Random) -> dict:
    kind = r.choice(["payment", "traffic", "funnel", "funnel2", "app_bug", "price", "stockout", "mix"])
    if kind == "payment":  # upi/cod only exist in IN, so pair them with a dimension that keeps IN
        pm = r.choice(["upi", "card", "card", "wallet"])
        f = _segment(r, ["os", "platform", "device_type", "category"] + ([] if pm == "upi" else ["country"]))
        return {"type": "payment_failure_spike", "filters": {**f, "payment_method": pm},
                "params": {"failure_rate_to": round(r.uniform(0.5, 0.9), 2)}}
    if kind == "traffic":
        return {"type": "traffic_drop", "params": {"traffic_factor": round(r.uniform(0.1, 0.6), 2)},
                "filters": _segment(r, ["traffic_source", "country", "platform", "device_type",
                                        "new_vs_returning"])}
    if kind == "funnel":
        return {"type": "funnel_drop", "params": {r.choice(FUNNEL): round(r.uniform(0.1, 0.55), 2)},
                "filters": _segment(r, ["country", "platform", "category", "traffic_source", "device_type"])}
    if kind == "funnel2":
        a, b = r.sample(["platform", "country", "category", "traffic_source", "new_vs_returning"], 2)
        return {"type": "funnel_drop", "params": {r.choice(FUNNEL): round(r.uniform(0.05, 0.4), 2)},
                "filters": {a: r.choice(VALUES[a]), b: r.choice(VALUES[b])}}
    if kind == "app_bug":
        return {"type": "app_version_bug", "params": {r.choice(FUNNEL): round(r.uniform(0.1, 0.5), 2)},
                "filters": {"os": r.choice(["android", "ios"]), "app_version": r.choice(["5.2.0", "5.3.0"])}}
    if kind == "price":
        m = r.choice([r.uniform(0.55, 0.8), r.uniform(1.25, 1.5)])
        return {"type": "pricing_change", "params": {"amount_mult": round(m, 2)},
                "filters": {"category": r.choice(VALUES["category"])}}
    if kind == "stockout":
        return {"type": "stockout", "params": {"atc_mult": round(r.uniform(0.05, 0.3), 2)},
                "filters": {"category": r.choice(VALUES["category"])}}
    return {"type": "mix_shift", "params": {"traffic_factor": round(r.uniform(2.0, 3.5), 1)},
            "filters": {"traffic_source": r.choice(["paid", "social"])}}


def random_scenarios(seed: int, days: int = 180) -> list[dict]:
    r = random.Random(seed)
    st = load_settings()["simulator_extra"]
    start, sale = st["start_date"], set(st["sale_days"])
    out = []
    for i, off in enumerate(range(35, days - 1, 5)):
        day = start + timedelta(days=off)
        if day in sale:
            continue
        effects = [_effect(r)]
        if r.random() < 0.2:  # two simultaneous causes on different dimensions
            e2 = _effect(r)
            if not set(e2["filters"]) & set(effects[0]["filters"]):
                effects.append(e2)
        out.append({"id": f"R{seed}_{i:02d}", "date": day, "type": "composite" if len(effects) > 1 else
                    effects[0]["type"], "category": "business", "effects": effects,
                    "expected_filters": [e["filters"] for e in effects], "expected_dq_status": "PASS",
                    "expected_detect": True, "expected_root_cause": ", ".join(e["type"] for e in effects)})
    return out


def make_dataset(seed: int, root: Path = Path("data/train")) -> Path:
    """Simulate one training dataset (~15 s). Skips work if it already exists."""
    out = root / f"seed_{seed}"
    if (out / "ground_truth.json").exists():
        return out
    scen = out / "_scenarios"
    scen.mkdir(parents=True, exist_ok=True)
    (scen / "random.yaml").write_text(yaml.safe_dump(random_scenarios(seed), sort_keys=False), encoding="utf-8")
    simulate(str(scen), str(out), None, None, seed, False)
    return out
