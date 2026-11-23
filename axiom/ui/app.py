"""Streamlit UI: pick a run, read what happened and why in plain words, open the evidence if you want proof.

Run with `axiom ui` (or `streamlit run axiom/ui/app.py`).
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Axiom", page_icon="🔎", layout="wide")
RUNS, EVALS = Path("runs"), Path("eval/results")
ICON = {"normal": "🟢", "anomaly": "🔴", "data_issue": "🟠"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_dirs() -> list[Path]:
    ds = [d for d in RUNS.glob("*") if (d / "report.json").exists()] if RUNS.exists() else []
    return sorted(ds, key=lambda d: (d / "report.json").stat().st_mtime, reverse=True)


def seg(filters: dict) -> str:
    return ", ".join(f"{k} = {v}" for k, v in filters.items()) or "the whole business"


def money(x) -> str:
    return "–" if x is None else f"{x:,.0f}"


def show_evidence(ev: dict):
    """One evidence record as a readable sentence or table; raw JSON only for tools without a view."""
    r, p, tool = ev["result"], ev["params"], ev["tool"]
    if tool == "drill_down" and "segments" in r:
        st.caption(f"{p.get('metric')} split by **{p.get('dimension')}**"
                   + (f" within {seg(p['filters'])}" if p.get("filters") else ""))
        df = pd.DataFrame(r["segments"][:8])
        df = pd.DataFrame({p.get("dimension", "segment"): df.segment, "normal day": df.baseline.round(2),
                           "this day": df.value.round(2), "share of the change": (df.share * 100).round(1)})
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif tool == "decompose_metric":
        st.caption(f"{p.get('metric')} broken into its parts")
        rows = [{"part": c["metric"], "normal day": round(c["baseline"], 4), "this day": round(c["value"], 4),
                 "share of the change": round(100 * c["share"], 1)} for c in r.get("children", [])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True) if rows else st.write(
            f"{p.get('metric')}: {r.get('value'):,.2f} vs {r.get('baseline'):,.2f} normally")
    elif tool == "segment_contribution":
        st.write(f"Revenue in **{seg(p['filters'])}** changed by {money(r['revenue_delta_in_segment'])} out of a "
                 f"total change of {money(r['revenue_delta_total'])}: **{100 * r['share']:.1f}%** of it.")
    elif tool == "check_traffic":
        st.write(f"Sessions in {seg(p.get('filters') or {})}: {money(r['sessions'])} vs "
                 f"{money(r['baseline'])} normally ({'normal' if r['within_normal'] else 'unusual'}).")
    elif tool == "detect":
        st.write(f"**{p['metric']}** was {r['value']:,.2f} vs {r['baseline']:,.2f} normally "
                 f"({100 * r['pct_change']:+.1f}%, z = {r['z']:.1f}).")
    else:
        st.json(r, expanded=False)


def report_page(d: Path):
    rep, ev = load(d / "report.json"), {e["evidence_id"]: e for e in (load(d / "evidence.json") or [])}
    day, status = rep["date"], rep["status"]

    if status == "normal":
        st.header(f"🟢 All normal on {day}")
        st.write("No key metric moved beyond its usual range. Nothing to do.")
    elif status == "data_issue":
        st.header(f"🟠 Data problem on {day}")
        st.write("The data for this day looks broken, so Axiom did not analyse it (bad data gives wrong answers).")
        for i in (load(d / "dq.json") or {}).get("issues", []):
            st.write(f"- **{i['check']}**: {i['detail']}")
    else:
        st.header(f"🔴 {rep['headline']}")
        st.caption(day)

    if status != "normal":
        a, b, c = st.columns(3)
        a.metric("Confidence", f"{rep['confidence']:.0%}")
        if rep.get("primary_cause"):
            b.metric("Explained", f"{rep['primary_cause']['contribution_pct']:.0f}%")
        c.metric("Alert", "Yes" if rep["escalate"] else "No")

    h = rep.get("primary_cause")
    if h:
        st.subheader("Why")
        st.success(f"**{h['statement']}**\n\nWhere: {seg(h['segment_filters'])}")
        for s in rep.get("secondary_causes", []):
            st.info(f"Also: {s['statement']} ({s['contribution_pct']:.0f}% of the change)")
        if rep.get("unexplained_pct", 0) >= 1:
            st.caption(f"{rep['unexplained_pct']:.0f}% of the change is not explained by any cause.")
    if status != "normal":
        st.subheader("What to do")
        st.write(rep["recommended_action"])

    if h:
        with st.expander("🔍 Show the proof (every number comes from here)"):
            for eid in h["evidence_ids"]:
                if eid in ev:
                    st.markdown(f"**{eid}**")
                    show_evidence(ev[eid])
    if rep.get("rejected"):
        with st.expander("❌ Ideas that were checked and rejected"):
            for r in rep["rejected"]:
                st.write(f"- {r['hypothesis']['statement']} — {'; '.join(r['reasons'])}")
    with st.expander("⚙️ Technical details"):
        st.markdown((d / "report.md").read_text(encoding="utf-8"))
        for name in ("dq", "detection", "investigation", "critic", "meta"):
            if (blob := load(d / f"{name}.json")) is not None:
                st.caption(name)
                st.json(blob, expanded=False)


def benchmark_page():
    st.header("Benchmark")
    st.write("How often each system finds the right cause on the simulator's 28 known incidents.")
    evs = sorted((d for d in EVALS.glob("*") if (d / "results.csv").exists()), reverse=True) if EVALS.exists() else []
    if not evs:
        st.info("No benchmark yet. Try `axiom eval --systems stats_only`.")
        return
    from axiom.eval import summarize

    s = summarize(pd.read_csv(st.selectbox("Benchmark run", evs, format_func=lambda p: p.name) / "results.csv"))
    view = pd.DataFrame({"right cause first": s["acc@1"], "right cause in top 3": s["acc@3"],
                         "catches real problems": s["det_recall"], "false alarms": s["false_positive_rate"],
                         "made-up numbers": s["numeric_hallucination"], "runs": s["runs"]})
    st.dataframe(view.round(2), use_container_width=True)
    with st.expander("All metrics"):
        st.dataframe(s.round(3), use_container_width=True)


st.sidebar.title("🔎 Axiom")
st.sidebar.caption("Finds out why your sales moved.")
page = st.sidebar.radio("View", ["Reports", "Benchmark"], label_visibility="collapsed")
if page == "Benchmark":
    benchmark_page()
else:
    runs = run_dirs()
    if not runs:
        st.info("No reports yet. Try `axiom run 2026-04-01`.")
    else:
        labels = {}
        for d in runs:
            r = load(d / "report.json")
            labels[d] = f"{ICON.get(r['status'], '⚪')} {r['date']} · {d.name}"
        pick = st.sidebar.radio("Reports (newest first)", runs, format_func=labels.get)
        report_page(pick)
