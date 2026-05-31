"""
app.py
======
Streamlit web application for FraudGuard AI.

Tabs:
  📊 Dashboard   — Live transaction feed with risk indicators
  🔍 Investigate — Select a transaction and run the agent
  🧠 Reasoning Log — View the end-to-end analysis of a transaction
  📋 Actions Log  — Durable audit trail of every committed protective action
  📐 Architecture  — System Architecture diagram
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# Limit numeric thread pool sizes — must be set before torch/numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import dotenv_values, load_dotenv

import torch
torch.classes.__path__ = []  # suppress spurious TorchScript warnings

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

from backend.database import (  # noqa: E402
    init_db,
    log_action,
    unfreeze_account,
    get_frozen_accounts,
    get_all_actions,
    get_flagged_transactions,
)
init_db()

# Prevent HuggingFace fast-tokenizer workers from leaking semaphores on shutdown.
# Must be set before sentence-transformers is imported anywhere.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Logs directory ────────────────────────────────────────────────────────────
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Check whether a valid API key is already in the .env file
_env_file = Path(__file__).parent / ".env"
_env_key  = dotenv_values(_env_file).get("OPENROUTER_API_KEY", "") if _env_file.exists() else ""
_KEY_IN_ENV = bool(_env_key) and not _env_key.startswith("your_")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 20px;
        font-weight: 600; border-radius: 8px 8px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ FraudGuard AI")
st.markdown(
    "An Intelligent Fraud Mitigation Platform\u2002·\u2002"
    "LLM: nvidia/nemotron-3-super-120b-a12b:free (OpenRouter)\u2002·\u2002"
    "ML: IsolationForest\u2002·\u2002Memory: FAISS"
)

# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Transaction Data…")
def load_data():
    from data.mock_data import ACCOUNTS, KNOWN_FRAUD_PATTERNS, generate_transactions
    return generate_transactions(), ACCOUNTS, KNOWN_FRAUD_PATTERNS


@st.cache_resource(show_spinner="Training ML Anomaly Detector…")
def load_detector(_df):
    from ml.anomaly_detector import FraudDetector
    det = FraudDetector()
    det.train(_df)
    return det


@st.cache_resource(show_spinner="Building Fraud Pattern Memory (downloading embeddings on first run)…")
def load_memory(_patterns):
    from agent.memory_store import FraudVectorMemory
    mem = FraudVectorMemory()
    mem.build_index(_patterns)
    return mem


# ── Initialise resources ──────────────────────────────────────────────────────
df, ACCOUNTS, KNOWN_FRAUD_PATTERNS = load_data()
detector = load_detector(df)
memory   = load_memory(KNOWN_FRAUD_PATTERNS)

# ── Session state ─────────────────────────────────────────────────────────────
if "action_log" not in st.session_state:
    st.session_state.action_log = []
if "investigation" not in st.session_state:
    st.session_state.investigation = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENROUTER_API_KEY", "")
if "show_sample_log" not in st.session_state:
    st.session_state.show_sample_log = False
if "pending_actions" not in st.session_state:
    st.session_state.pending_actions = []
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "hitl_decision" not in st.session_state:
    st.session_state.hitl_decision = None


# ── Investigation log helpers ────────────────────────────────────────────────

def _build_log_trace(reasoning_steps: list) -> list:
    """Convert agent reasoning_steps → sample-log-compatible trace entries."""
    trace = []
    step_num = 0
    for rs in reasoning_steps:
        step_num += 1
        reasoning = rs.get("reasoning", "").strip()
        if reasoning:
            trace.append({"step": step_num, "type": "reasoning", "content": reasoning})
        raw_thought = rs.get("thought", "")
        thought = (
            raw_thought
            .replace("Thought:", "")
            .split("\nAction:")[0]
            .strip()
        )
        if thought:
            trace.append({"step": step_num, "type": "thought", "content": thought})
        action = rs.get("action", "")
        if action:
            trace.append({
                "step": step_num,
                "type": "action",
                "tool": action,
                "input": str(rs.get("action_input", "")),
                "output": str(rs.get("observation", "")),
            })
    return trace


def _extract_fraud_type(final_answer: str) -> str:
    return _parse_final_report(final_answer)["fraud_type"]


def _parse_final_report(text: str) -> dict:
    """Extract structured sections from the agent's final report text."""

    def _first(pattern: str, t: str, default: str = "N/A") -> str:
        m = re.search(pattern, t, re.IGNORECASE)
        return m.group(1).strip() if m else default

    def _section_list(header_pat: str, stop_pat: str, t: str) -> list[str]:
        m = re.search(
            header_pat + r"[:\s]*\n(.*?)(?=" + stop_pat + r"|\Z)",
            t, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return []
        items = []
        for ln in m.group(1).splitlines():
            ln = ln.strip().lstrip("-•* \t").strip()
            if ln and ln != "---":
                items.append(ln)
        return items

    # Strip surrounding --- delimiters the agent sometimes adds
    text = re.sub(r"^---\s*", "", text.strip())
    text = re.sub(r"\s*---$", "", text)

    verdict    = _first(r"\*{0,2}VERDICT\*{0,2}:?\s*(\w+)", text)
    risk_score = _first(r"\*{0,2}Risk Score\*{0,2}:?\s*([\d /]+)", text)
    fraud_type = _first(r"\*{0,2}Fraud Type\*{0,2}:?\s*([^\n]+)", text)
    risk_factors = _section_list(
        r"\*{0,2}Key Risk Factors\*{0,2}",
        r"\*{0,2}Actions Taken\*{0,2}",
        text,
    )
    actions = _section_list(
        r"\*{0,2}Actions Taken\*{0,2}",
        r"\*{0,2}Recovery Steps",
        text,
    )
    recovery = _section_list(
        r"\*{0,2}Recovery Steps for Customer\*{0,2}",
        r"---",
        text,
    )
    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "fraud_type": fraud_type,
        "risk_factors": risk_factors,
        "actions": actions,
        "recovery": recovery,
        "_raw": text,
    }


def _clean(text: str) -> str:
    """Strip markdown inline markers and escape $ to prevent LaTeX rendering."""
    text = re.sub(r"[*_`]", "", text).strip()
    text = text.replace("$", r"\$")
    return text


def _render_final_report(text: str) -> None:
    """Parse the agent's structured final report and render it using native Streamlit."""
    if not text:
        st.warning("No report generated.")
        return

    parsed = _parse_final_report(text)

    # Fall back to raw markdown if parsing found nothing meaningful
    if parsed["verdict"] == "N/A" and not parsed["risk_factors"]:
        st.markdown(text)
        return

    verdict = parsed["verdict"].upper() if parsed["verdict"] != "N/A" else "UNKNOWN"
    v_icon  = {"FRAUDULENT": "🔴", "SUSPICIOUS": "🟡", "LEGITIMATE": "🟢"}.get(verdict, "⚪")
    ft_display = parsed["fraud_type"].replace("_", " ").title() if parsed["fraud_type"] != "N/A" else "N/A"

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict_line = f"{v_icon} **{verdict}**  ·  Risk Score: {parsed['risk_score']}  ·  Fraud Type: {ft_display}"
    if verdict == "FRAUDULENT":
        st.error(verdict_line)
    elif verdict == "SUSPICIOUS":
        st.warning(verdict_line)
    else:
        st.success(verdict_line)

    # ── Key Risk Factors ──────────────────────────────────────────────────────
    if parsed["risk_factors"]:
        st.markdown("**🚨 Key Risk Factors**")
        for factor in parsed["risk_factors"]:
            st.write(f"• {_clean(factor)}")

    st.divider()

    # ── Actions Taken | Recovery Steps ───────────────────────────────────────
    col_a, col_r = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**⚡ Actions Taken**")
            if parsed["actions"]:
                for action in parsed["actions"]:
                    st.write(f"• {_clean(action)}")
            else:
                st.write("None required")
    with col_r:
        with st.container(border=True):
            st.markdown("**💡 Recovery Steps for Customer**")
            if parsed["recovery"]:
                for step in parsed["recovery"]:
                    st.write(f"• {_clean(step)}")
            else:
                st.write("No steps provided")


def save_investigation_log(
    result: dict,
    txn_df,
    accounts: dict,
    committed_actions: list | None = None,
    hitl_decision: dict | None = None,
) -> tuple[Path, str]:
    """Persist an investigation result to logs/ as a JSON file.

    Returns (path, investigation_id).
    """
    tid = result.get("transaction_id", "UNKNOWN")
    row = txn_df[txn_df["transaction_id"] == tid]
    account_id = row.iloc[0]["account_id"] if not row.empty else "UNKNOWN"
    acc = accounts.get(account_id, {})

    now = datetime.now(timezone.utc)
    investigation_id = f"INV-{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond // 1000:03d}"
    log_entry = {
        "investigation_id": investigation_id,
        "transaction_id": tid,
        "account_id": account_id,
        "account_holder": acc.get("name", "Unknown"),
        "investigated_at": now.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "model_used": "nvidia/nemotron-3-super-120b-a12b:free",
        "verdict": result.get("verdict", "UNKNOWN"),
        "risk_score": int(result["risk_score"]) if result.get("risk_score") is not None else None,
        "amount": round(float(row.iloc[0]["amount"]), 2) if not row.empty else None,
        "fraud_type": _extract_fraud_type(result.get("final_answer", "")),
        "elapsed_seconds": round(float(result["elapsed_seconds"]), 2) if result.get("elapsed_seconds") is not None else None,
        "reasoning_trace": _build_log_trace(result.get("reasoning_steps", [])),
        "final_report": result.get("final_answer", ""),
        "committed_actions": committed_actions or [],
        "hitl_decision": hitl_decision or {},
        "error": result.get("error"),
        "error_type": result.get("error_type"),
    }

    safe_tid = re.sub(r"[^A-Za-z0-9_-]", "", tid)
    filename = LOGS_DIR / f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_tid}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2)
    return filename, investigation_id


def list_saved_logs() -> list[Path]:
    """Return saved log files sorted newest-first."""
    return sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def render_log(log_data: dict) -> None:
    """Render a log entry (works for both saved real logs and the sample log)."""
    acc_line = ""
    if log_data.get("account_holder"):
        amount_str = f"  ·  ${log_data['amount']:,.2f}" if log_data.get("amount") is not None else ""
        acc_line = (
            f"**Account**: {log_data.get('account_id')} — {log_data.get('account_holder')}{amount_str}  \n"
        )
    if log_data.get("fraud_type") and log_data["fraud_type"] != "N/A":
        acc_line += f"**Fraud Type**: {log_data['fraud_type']}  \n"
    if log_data.get("elapsed_seconds"):
        acc_line += f"**Duration**: {log_data['elapsed_seconds']}s  \n"
    if log_data.get("investigated_at"):
        acc_line += f"**Investigated at**: {log_data['investigated_at']}"
    if acc_line:
        st.markdown(acc_line)

    st.divider()

    trace = log_data.get("reasoning_trace", [])
    if trace:
        with st.expander(f"🧠 Reasoning Trace ({len([s for s in trace if s.get('type')=='action'])} tool calls)", expanded=True):
            for step in trace:
                step_type = step.get("type", "thought")
                if step_type == "reasoning":
                    st.markdown(f"**🧠 Model Reasoning — step {step['step']}**")
                    st.write(_clean(step["content"]))
                elif step_type == "thought":
                    st.markdown(f"**Step {step['step']} · Thought**")
                    st.write(_clean(step["content"]))
                elif step_type == "action":
                    st.markdown(f"**🔧 Tool Invoked**: `{step['tool']}`")
                    st.markdown("**Input**")
                    st.code(str(step['input']), language=None)
                    st.markdown("**Output**")
                    st.code(step["output"], language=None)
                    st.divider()

    st.divider()
    st.subheader("📄 Final Report")
    with st.container(border=True):
        _render_final_report(log_data.get("final_report", ""))

    if log_data.get("error"):
        st.error(f"Agent error during investigation: {log_data['error']}")

    with st.expander("📦 Raw JSON"):
        st.json(log_data)


# ── Enrich df with ML scores ──────────────────────────────────────────────────
@st.cache_data
def enrich_df(_df, _detector):
    scores, levels, flags = [], [], []
    for tid in _df["transaction_id"]:
        r = _detector.get_risk_assessment(tid)
        scores.append(r["risk_score"])
        levels.append(r["risk_level"])
        flags.append(", ".join(r["anomaly_flags"]) if r["anomaly_flags"] else "—")
    out = _df.copy()
    out["risk_score"] = scores
    out["risk_level"] = levels
    out["anomaly_flags"] = flags
    return out


enriched = enrich_df(df, detector)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_investigate, tab_log, tab_actions, tab_arch = st.tabs(
    ["📊 Dashboard", "🔍 Investigate", "🧠 Reasoning Log", "📋 Actions Log", "📐 Architecture"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dash:
    total       = len(enriched)
    flagged     = int((enriched["risk_score"] >= 35).sum())
    fraud_det   = int((enriched["risk_score"] >= 70).sum())
    amount_risk = enriched[enriched["risk_score"] >= 70]["amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total)
    c2.metric("Flagged for Review", flagged)
    c3.metric("High-Risk (ML)", fraud_det)
    c4.metric("Amount at Risk", f"${amount_risk:,.0f}")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Transaction Feed")
        show_df = enriched[
            ["transaction_id", "account_id", "timestamp", "amount", "merchant",
             "location", "risk_score", "risk_level", "anomaly_flags"]
        ].copy()
        show_df["timestamp"] = show_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        show_df = show_df.sort_values("risk_score", ascending=False).head(50)

        st.dataframe(
            show_df.rename(columns={
                "transaction_id": "TXN ID",
                "account_id": "Account",
                "timestamp": "Time",
                "amount": "Amount ($)",
                "merchant": "Merchant",
                "location": "Location",
                "risk_score": "Risk Score",
                "risk_level": "Level",
                "anomaly_flags": "Flags",
            }),
            width='stretch',
            height=420,
            column_config={
                "Amount ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Risk Score": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%d"
                ),
            },
            hide_index=True,
        )

    with col_right:
        st.subheader("Risk Distribution")
        bins = pd.cut(
            enriched["risk_score"],
            bins=[0, 35, 70, 100],
            labels=["Low (0–34)", "Medium (35–69)", "High (70–100)"],
        )
        dist = bins.value_counts().reset_index()
        dist.columns = ["Level", "Count"]
        fig = px.pie(
            dist, values="Count", names="Level",
            color="Level",
            color_discrete_map={
                "Low (0–34)": "#27ae60",
                "Medium (35–69)": "#f39c12",
                "High (70–100)": "#e74c3c",
            },
            hole=0.45,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width='stretch')

        st.subheader("Top Fraud Accounts")
        acc_fraud = (
            enriched[enriched["risk_score"] >= 70]
            .groupby("account_id")["risk_score"]
            .count()
            .reset_index()
            .rename(columns={"risk_score": "High-Risk Txns"})
            .sort_values("High-Risk Txns", ascending=False)
        )
        acc_fraud["Account Name"] = acc_fraud["account_id"].map(
            lambda a: ACCOUNTS.get(a, {}).get("name", a)
        )
        st.dataframe(
            acc_fraud[["Account Name", "account_id", "High-Risk Txns"]],
            hide_index=True, width='stretch', height=200
        )

    # Timeline chart
    st.subheader("Transaction Timeline — Amounts by Risk Level")
    timeline_df = enriched.copy()
    timeline_df["Risk Category"] = pd.cut(
        timeline_df["risk_score"],
        bins=[0, 35, 70, 100],
        labels=["Low", "Medium", "High"],
    )
    fig2 = px.scatter(
        timeline_df,
        x="timestamp", y="amount",
        color="Risk Category",
        color_discrete_map={"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"},
        hover_data=["transaction_id", "merchant", "account_id"],
        size="amount",
        size_max=20,
        opacity=0.75,
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,37,64,0.8)",
        font_color="white",
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", title="Amount ($)"),
        height=320,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: INVESTIGATE
# ─────────────────────────────────────────────────────────────────────────────
with tab_investigate:
    st.subheader("🔍 AI Agent-Powered Transaction Investigation")

    # API key input — only shown when the key is not already set in .env
    if not _KEY_IN_ENV:
        api_key_input = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.api_key,
            type="password",
            help="Set OPENROUTER_API_KEY in your .env file, or paste it here.",
        )
        if api_key_input:
            st.session_state.api_key = api_key_input
    else:
        st.session_state.api_key = _env_key

    api_ok = bool(st.session_state.api_key) and not st.session_state.api_key.startswith("your_")
    if not api_ok:
        st.warning("Enter your OpenRouter API key above to enable live agent investigation.")

    st.divider()
    st.markdown("#### Choose Transaction")

    sort_opts = {
        "Risk Score ↓": ("risk_score", False),
        "Amount ↓":     ("amount", False),
        "Timestamp ↑":  ("timestamp", True),
        "Timestamp ↓":  ("timestamp", False),
    }
    sort_col_choice, _ = st.columns([1, 3])
    with sort_col_choice:
        sort_choice = st.selectbox("Sort by", list(sort_opts.keys()), index=0, key="txn_sort")
    sort_col, sort_asc = sort_opts[sort_choice]
    sorted_enriched = enriched.sort_values(sort_col, ascending=sort_asc)
    all_ids = sorted_enriched["transaction_id"].tolist()

    # Pre-build lookup for fast format_func (avoids O(n) scan per dropdown item)
    _txn_meta = {
        row["transaction_id"]: (
            pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            row["amount"],
            int(row["risk_score"]),
        )
        for _, row in sorted_enriched.iterrows()
    }

    selected_tid = st.selectbox(
        "Select transaction",
        options=all_ids,
        format_func=lambda t: (
            f"{_txn_meta[t][0]}  |  {t}  |  ${_txn_meta[t][1]:,.2f}  |  Risk score: {_txn_meta[t][2]}/100"
        ),
    )

    if selected_tid:
        row = enriched[enriched["transaction_id"] == selected_tid].iloc[0]
        r = detector.get_risk_assessment(selected_tid)
        acc = ACCOUNTS.get(row["account_id"], {})

        with st.container(border=True):
            st.markdown(f"**Transaction Preview** — `{selected_tid}`")
            preview_cols = st.columns(3)
            preview_cols[0].metric("Amount", f"${row['amount']:,.2f}")
            preview_cols[1].metric("ML Risk Score", f"{r['risk_score']}/100")
            preview_cols[2].metric("Risk Level", r["risk_level"])

            st.markdown(
                f"**Merchant**: {row['merchant']}  \n"
                f"**Category**: {row['category']}  \n"
                f"**Location**: {row['location']}  \n"
                f"**Account**: {row['account_id']} — {acc.get('name', '')}  \n"
                f"**Timestamp**: {pd.Timestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M')}"
            )
            if r["anomaly_flags"]:
                st.markdown(
                    "**Anomaly Flags**: "
                    + "  ".join(f'`{f}`' for f in r["anomaly_flags"])
                )

    st.divider()

    investigate_btn = st.button(
        "🚀 Run Agent Investigation",
        type="primary",
        disabled=not api_ok,
        width='stretch',
    )

    if investigate_btn and api_ok and selected_tid:
        from agent.tools import create_fraud_tools
        from agent.fraud_agent import FraudMitigationAgent

        st.session_state.action_log = []
        st.session_state.pending_actions = []
        st.session_state.hitl_pending = False
        st.session_state.hitl_decision = None
        st.session_state.show_sample_log = False

        with st.spinner("Agent is investigating — this may take some time…"):
            tools = create_fraud_tools(
                transactions_df=df,
                accounts=ACCOUNTS,
                detector=detector,
                memory=memory,
                pending_actions=st.session_state.pending_actions,
            )
            agent = FraudMitigationAgent(
                api_key=st.session_state.api_key, tools=tools
            )
            result = agent.investigate(selected_tid)
            st.session_state.investigation = result

            _verdict = result.get("verdict", "UNKNOWN")
            _freeze_queued = any(
                a["action"] == "FREEZE_ACCOUNT"
                for a in st.session_state.pending_actions
            )
            if _verdict == "SUSPICIOUS" or (_verdict == "FRAUDULENT" and _freeze_queued):
                st.session_state.hitl_pending = True
            else:
                for _a in st.session_state.pending_actions:
                    st.session_state.action_log.append(
                        {k: v for k, v in _a.items() if k != "status"}
                    )
                st.session_state.pending_actions = []
                _sp, _inv_id = save_investigation_log(
                    result, df, ACCOUNTS,
                    committed_actions=st.session_state.action_log,
                    hitl_decision=None,
                )
                for _a in st.session_state.action_log:
                    log_action(
                        entry=_a,
                        analyst_decision="auto",
                        analyst_notes="",
                        risk_score=result.get("risk_score"),
                        agent_verdict=result.get("verdict", "UNKNOWN"),
                        investigation_id=_inv_id,
                    )
                if _sp:
                    st.toast(f"Log saved → {_sp.name}", icon="💾")

    # ── Display investigation result ─────────────────────────────────────────
    inv = st.session_state.investigation
    if inv and inv.get("transaction_id") == selected_tid:
        st.subheader("Investigation Result")

        # ── API / agent error handling ────────────────────────────────────────
        if inv.get("error"):
            _error_labels = {
                "auth_error":      "API key is invalid or not authorised. Check your OPENROUTER_API_KEY.",
                "rate_limit":      "Rate limit reached — too many requests. Wait a moment and try again.",
                "quota_exhausted": "Daily free-tier quota on OpenRouter is exhausted. Try again tomorrow.",
                "connection_error": "Cannot reach the OpenRouter API. Check your internet connection.",
            }
            _etype = inv.get("error_type", "unknown_error")
            st.error(_error_labels.get(_etype, f"Agent error: {inv['error']}"))
            if st.button("Show sample investigation log", key="show_sample_btn"):
                st.session_state.show_sample_log = True

        if st.session_state.get("show_sample_log") and inv.get("error"):
            sample_path = Path(__file__).parent / "sample_reasoning_log.json"
            if sample_path.exists():
                with open(sample_path, encoding="utf-8") as _sf:
                    sample_data = json.load(_sf)
                st.info("Showing a pre-recorded sample investigation log.")
                render_log(sample_data)

        if not inv.get("error"):
            # Verdict banner
            verdict = inv.get("verdict", "UNKNOWN")
            risk_s  = inv.get("risk_score")
            elapsed = inv.get("elapsed_seconds", 0)

            verdict_icon = {"FRAUDULENT": "🔴", "SUSPICIOUS": "🟡", "LEGITIMATE": "🟢"}.get(verdict, "⚪")
            verdict_msg  = f"{verdict_icon} **{verdict}** — Risk Score: {risk_s}/100 · Completed in {elapsed}s"
            if verdict == "FRAUDULENT":
                st.error(verdict_msg)
            elif verdict == "SUSPICIOUS":
                st.warning(verdict_msg)
            else:
                st.success(verdict_msg)

            # Reasoning steps
            steps = inv.get("reasoning_steps", [])
            if steps:
                with st.expander(f"🧠 Agent Reasoning Trace ({len(steps)} steps)", expanded=True):
                    for i, step in enumerate(steps, 1):
                        reasoning = step.get("reasoning", "").strip()
                        thought   = step.get("thought", "").strip()
                        action    = step.get("action", "")
                        act_in    = step.get("action_input", "")
                        obs       = step.get("observation", "")

                        if reasoning:
                            st.markdown(f"**🧠 Model Reasoning (step {i})**")
                            st.write(_clean(reasoning))
                        if thought:
                            clean_thought = thought.replace("Thought:", "").split("\nAction:")[0].strip()
                            if clean_thought:
                                st.markdown(f"**Step {i} · Thought**")
                                st.write(_clean(clean_thought))
                        if action:
                            st.markdown(f"**🔧 Tool Invoked**: `{action}`")
                            st.markdown("**Input**")
                            st.code(str(act_in), language=None)
                        if obs:
                            st.markdown("**Output**")
                            st.code(str(obs), language=None)
                            st.divider()

            # Final structured report
            st.subheader("📄 Final Report")
            with st.container(border=True):
                _render_final_report(inv.get("final_answer", ""))

            # ── Human-in-the-loop panel ──────────────────────────────────────────
            _verdict_now = inv.get("verdict", "UNKNOWN")
            if st.session_state.get("hitl_pending"):
                st.divider()
                if _verdict_now == "FRAUDULENT":
                    # ── Touchpoint 1: Confirm account freeze ──────────────────
                    _freeze_items = [
                        a for a in st.session_state.pending_actions
                        if a["action"] == "FREEZE_ACCOUNT"
                    ]
                    st.subheader("🔐 Human Approval Required — Account Freeze")
                    st.error(
                        "The agent recommends freezing the account below. "
                        "**This action is irreversible.** Review the full report above before deciding."
                    )
                    for _a in _freeze_items:
                        with st.container(border=True):
                            st.markdown(
                                f"**Account**: `{_a.get('account_id', '?')}` — {_a.get('account_holder', 'Unknown')}  \n"
                                f"**Reason**: {_a.get('reason', '—')}  \n"
                                f"**Queued at**: {_a.get('timestamp', '—')}"
                            )
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        if st.button(
                            "✅ Approve — Execute Freeze",
                            type="primary",
                            key="hitl_approve_freeze",
                            width='stretch',
                        ):
                            _ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            for _a in st.session_state.pending_actions:
                                st.session_state.action_log.append(
                                    {k: v for k, v in _a.items() if k != "status"}
                                )
                            st.session_state.pending_actions = []
                            st.session_state.hitl_pending = False
                            st.session_state.hitl_decision = {"decision": "approved", "timestamp": _ts}
                            _sp, _inv_id = save_investigation_log(
                                inv, df, ACCOUNTS,
                                committed_actions=st.session_state.action_log,
                                hitl_decision=st.session_state.hitl_decision,
                            )
                            for _a in st.session_state.action_log:
                                log_action(
                                    entry=_a,
                                    analyst_decision="approved",
                                    analyst_notes="",
                                    risk_score=inv.get("risk_score"),
                                    agent_verdict=inv.get("verdict", "UNKNOWN"),
                                    investigation_id=_inv_id,
                                )
                            if _sp:
                                st.toast(f"Log saved → {_sp.name}", icon="💾")
                            st.rerun()
                    with _c2:
                        if st.button(
                            "❌ Reject Freeze — Flag Only",
                            key="hitl_reject_freeze",
                            width='stretch',
                        ):
                            _ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            for _a in st.session_state.pending_actions:
                                if _a["action"] == "FLAG_TRANSACTION":
                                    st.session_state.action_log.append(
                                        {k: v for k, v in _a.items() if k != "status"}
                                    )
                            st.session_state.pending_actions = []
                            st.session_state.hitl_pending = False
                            st.session_state.hitl_decision = {"decision": "rejected_freeze", "timestamp": _ts}
                            _sp, _inv_id = save_investigation_log(
                                inv, df, ACCOUNTS,
                                committed_actions=st.session_state.action_log,
                                hitl_decision=st.session_state.hitl_decision,
                            )
                            for _a in st.session_state.action_log:
                                log_action(
                                    entry=_a,
                                    analyst_decision="rejected_freeze",
                                    analyst_notes="",
                                    risk_score=inv.get("risk_score"),
                                    agent_verdict=inv.get("verdict", "UNKNOWN"),
                                    investigation_id=_inv_id,
                                )
                            if _sp:
                                st.toast(f"Log saved → {_sp.name}", icon="💾")
                            st.rerun()

                elif _verdict_now == "SUSPICIOUS":
                    # ── Touchpoint 2: Analyst review ──────────────────────────
                    st.subheader("👤 Analyst Review Required — SUSPICIOUS Transaction")
                    st.warning(
                        "The agent rated this transaction **SUSPICIOUS** (borderline risk 35–69). "
                        "No automated actions have been taken. "
                        "Review the evidence above, then submit your determination."
                    )
                    _analyst_verdict = st.radio(
                        "Your determination:",
                        options=["escalate", "monitor", "downgrade"],
                        format_func=lambda x: {
                            "escalate":  "🔴 Escalate to FRAUDULENT — flag transaction and freeze account",
                            "monitor":   "🟡 Monitor only — flag transaction, no account freeze",
                            "downgrade": "🟢 Downgrade to LEGITIMATE — no action required",
                        }[x],
                        key="hitl_analyst_verdict",
                    )
                    _analyst_notes = st.text_area(
                        "Analyst notes (optional)",
                        placeholder="e.g. Verified with customer — travel confirmed",
                        key="hitl_analyst_notes",
                    )
                    if st.button(
                        "Submit Decision",
                        type="primary",
                        key="hitl_submit_analyst",
                        width='stretch',
                    ):
                        _ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        _txn_r = df[df["transaction_id"] == inv["transaction_id"]]
                        _acc_id = _txn_r.iloc[0]["account_id"] if not _txn_r.empty else "UNKNOWN"
                        _acc_name = ACCOUNTS.get(_acc_id, {}).get("name", "Unknown")
                        _notes_text = _analyst_notes.strip() or "None"
                        if _analyst_verdict in ("escalate", "monitor"):
                            st.session_state.action_log.append({
                                "action": "FLAG_TRANSACTION",
                                "transaction_id": inv["transaction_id"],
                                "account_id": _acc_id,
                                "reason": f"Analyst {_analyst_verdict}. Notes: {_notes_text}",
                                "timestamp": _ts,
                            })
                        if _analyst_verdict == "escalate":
                            st.session_state.action_log.append({
                                "action": "FREEZE_ACCOUNT",
                                "account_id": _acc_id,
                                "account_holder": _acc_name,
                                "reason": f"Analyst escalation from SUSPICIOUS. Notes: {_notes_text}",
                                "timestamp": _ts,
                            })
                        st.session_state.hitl_pending = False
                        st.session_state.pending_actions = []
                        st.session_state.hitl_decision = {
                            "decision": _analyst_verdict,
                            "notes": _notes_text,
                            "timestamp": _ts,
                        }
                        _sp, _inv_id = save_investigation_log(
                            inv, df, ACCOUNTS,
                            committed_actions=st.session_state.action_log,
                            hitl_decision=st.session_state.hitl_decision,
                        )
                        for _a in st.session_state.action_log:
                            log_action(
                                entry=_a,
                                analyst_decision=_analyst_verdict,
                                analyst_notes=_notes_text,
                                risk_score=inv.get("risk_score"),
                                agent_verdict=inv.get("verdict", "UNKNOWN"),
                                investigation_id=_inv_id,
                            )
                        if _sp:
                            st.toast(f"Log saved → {_sp.name}", icon="💾")
                        st.rerun()

            else:
                # ── Decision made (or LEGITIMATE/auto-commit path) ────────────
                _dec = st.session_state.get("hitl_decision")
                if _dec:
                    _dec_labels = {
                        "approved":        "✅ Analyst approved — account freeze executed",
                        "rejected_freeze": "🟡 Analyst rejected freeze — transaction flagged only",
                        "escalate":        "🔴 Analyst escalated to FRAUDULENT",
                        "monitor":         "🟡 Analyst flagged for monitoring (no freeze)",
                        "downgrade":       "🟢 Analyst downgraded to LEGITIMATE — no action taken",
                    }
                    _label = _dec_labels.get(_dec.get("decision", ""), "Decision recorded")
                    if _dec.get("decision") in ("approved", "escalate"):
                        st.error(_label)
                    elif _dec.get("decision") == "downgrade":
                        st.success(_label)
                    else:
                        st.warning(_label)
                    if _dec.get("notes") and _dec["notes"] != "None":
                        st.caption(f"Analyst notes: {_dec['notes']}")

                if st.session_state.action_log:
                    with st.expander(
                        f"🔒 Protective Actions ({len(st.session_state.action_log)})",
                        expanded=True,
                    ):
                        for entry in st.session_state.action_log:
                            _at = entry.get("action", "UNKNOWN")
                            if _at == "FLAG_TRANSACTION":
                                st.warning(
                                    f"🚩 **Transaction Flagged** — **{entry.get('transaction_id')}**  \n"
                                    f"Account: **{entry.get('account_id')}**  \n"
                                    f"Reason: {entry.get('reason')}  \n"
                                    f"At: {entry.get('timestamp')}"
                                )
                            elif _at == "FREEZE_ACCOUNT":
                                st.error(
                                    f"🔒 **Account Frozen** — **{entry.get('account_id')}** ({entry.get('account_holder')})  \n"
                                    f"Reason: {entry.get('reason')}  \n"
                                    f"At: {entry.get('timestamp')}"
                                )
                            else:
                                st.json(entry)

    elif inv and inv.get("transaction_id") != selected_tid:
        if st.session_state.get("hitl_pending"):
            st.warning(
                f"A HITL decision is still pending for **{inv.get('transaction_id')}**. "
                "Re-select that transaction to complete it, or run a new investigation to discard it."
            )
        else:
            st.info("Select a transaction and click 'Run Agent Investigation' to start.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: REASONING LOG
# ─────────────────────────────────────────────────────────────────────────────
with tab_log:
    st.subheader("🧠 Investigation Reasoning Logs")

    saved_logs = list_saved_logs()

    if saved_logs:
        # ── Real logs exist — let user pick one ───────────────────────────────
        st.markdown(
            f"**{len(saved_logs)} saved investigation(s)** found in `logs/`. "
            "Select one to inspect the agent's full reasoning trace."
        )

        # Build a human-readable label for each log
        def _log_label(p: Path) -> str:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                tid     = d.get("transaction_id", p.stem)
                verdict = d.get("verdict", "?")
                ts      = d.get("investigated_at", "")[:16].replace("T", " ")
                return f"{tid}  ·  {verdict}  ·  {ts}"
            except Exception:  # noqa: BLE001
                return p.name

        log_labels = [_log_label(p) for p in saved_logs]
        selected_label = st.selectbox(
            "Select investigation log",
            options=log_labels,
            index=0,
        )
        selected_log_path = saved_logs[log_labels.index(selected_label)]

        with open(selected_log_path, encoding="utf-8") as f:
            log_data = json.load(f)

        render_log(log_data)

    else:
        # ── No real logs yet ──────────────────────────────────────────────────
        st.info(
            "No investigation logs found yet. "
            "Run an investigation in the **🔍 Investigate** tab and the full "
            "reasoning trace will be saved here automatically."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ACTIONS LOG
# ─────────────────────────────────────────────────────────────────────────────
with tab_actions:
    st.subheader("📋 Actions Log")
    st.markdown(
        "Durable record of every protective action committed through the HITL panels. "
        "Persists across sessions in `backend/fraud_actions.db`."
    )

    # ── Currently Frozen Accounts ─────────────────────────────────────────────
    st.markdown("#### 🔒 Currently Frozen Accounts")
    _frozen = get_frozen_accounts()
    if _frozen:
        for _facc in _frozen:
            with st.container(border=True):
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"**{_facc['account_id']}** — {_facc['holder_name'] or 'Unknown'}  \n"
                        f"**Frozen since**: {_facc['frozen_at']}  \n"
                        f"**Reason**: {_facc['freeze_reason'] or '—'}"
                    )
                with col_btn:
                    if st.button(
                        "Unfreeze",
                        key=f"unfreeze_{_facc['account_id']}",
                        type="secondary",
                        width='stretch',
                    ):
                        unfreeze_account(_facc["account_id"])
                        st.toast(f"Account {_facc['account_id']} unfrozen", icon="✅")
                        st.rerun()
    else:
        st.success("No accounts currently frozen.")

    st.divider()

    # ── Flagged Transactions ──────────────────────────────────────────────────
    st.markdown("#### 🚩 Flagged Transactions")
    _flags = get_flagged_transactions()
    if _flags:
        _flags_df = pd.DataFrame(_flags)[
            ["transaction_id", "account_id", "reason", "analyst_decision", "agent_verdict",
             "risk_score", "created_at"]
        ].rename(columns={
            "transaction_id":   "TXN ID",
            "account_id":       "Account",
            "reason":           "Reason",
            "analyst_decision": "Decision",
            "agent_verdict":    "Agent Verdict",
            "risk_score":       "Risk Score",
            "created_at":       "Flagged At",
        })
        st.dataframe(
            _flags_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            },
        )
    else:
        st.info("No transactions flagged yet.")

    st.divider()

    # ── Full Audit Log ────────────────────────────────────────────────────────
    st.markdown("#### 🗂 Full Audit Log")
    _all = get_all_actions(200)
    if _all:
        _all_df = pd.DataFrame(_all)[
            ["created_at", "action_type", "transaction_id", "account_id", "account_holder",
             "agent_verdict", "risk_score", "analyst_decision", "analyst_notes",
             "reason", "investigation_id"]
        ].rename(columns={
            "created_at":       "Timestamp",
            "action_type":      "Action",
            "transaction_id":   "TXN ID",
            "account_id":       "Account",
            "account_holder":   "Holder",
            "agent_verdict":    "Agent Verdict",
            "risk_score":       "Risk",
            "analyst_decision": "Decision",
            "analyst_notes":    "Notes",
            "reason":           "Reason",
            "investigation_id": "Investigation",
        })
        st.dataframe(
            _all_df,
            width='stretch',
            hide_index=True,
            height=400,
            column_config={
                "Risk": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            },
        )
        st.caption(f"Showing the {len(_all)} most recent action(s) (limit 200).")
    else:
        st.info("No actions recorded yet. Run an investigation and commit an action to see it here.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
with tab_arch:
    st.subheader("📐 Agent Logic & Architecture")

    st.markdown(
        """
        The diagram below shows the full system: the **LLM brain**, the **10 tools** it can invoke,
        the three-component **memory** layer (in-context scratchpad, FAISS vector store, and
        the pending actions queue), the **HITL approval gate** that prevents autonomous account
        freezes, and the **SQLite persistence layer** that durably records every committed action.
        """
    )

    dot_src = """
digraph FraudGuard {
    rankdir=TB
    bgcolor="#0d1117"
    node [style="filled,rounded" fontcolor="white" fontname="Helvetica" shape=box penwidth=1.5 margin="0.2,0.1"]
    edge [arrowsize=0.7]

    USER   [label="User / Analyst" fillcolor="#1e3a5f" color="#4a9eff"]
    UI     [label="Streamlit Web UI\nDashboard  Investigate  Reasoning Log" fillcolor="#1e3a5f" color="#4a9eff"]
    HITL   [label="HITL Panel\nApprove Freeze / Reject / Analyst Review" fillcolor="#2a0a10" color="#e74c3c"]
    ACTLOG [label="Actions Log Tab\nFrozen accounts  Flagged txns  Audit trail" fillcolor="#1e3a5f" color="#4a9eff"]

    subgraph cluster_agent {
        label="FraudGuard Agent Brain"
        style="filled,rounded"  fillcolor="#0a1a30"  color="#4a9eff"  fontcolor="white"
        LLM  [label="LLM: nvidia/nemotron-3-super-120b-a12b\nOpenRouter API (free, 262K ctx)" fillcolor="#1a3060" color="#4a9eff"]
        LOOP [label="Tool-calling Loop\nSystem Prompt  ->  Call Tool  ->  Observe" fillcolor="#1a3060" color="#4a9eff"]
        LLM -> LOOP [style=dashed color="#3a6aaf"]
    }

    subgraph cluster_tools {
        label="Agent Tools (10)"
        style="filled,rounded"  fillcolor="#150a25"  color="#9b59b6"  fontcolor="white"
        T1  [label="get_transaction_details"     fillcolor="#251040" color="#9b59b6"]
        T2  [label="analyze_ml_risk_score"        fillcolor="#251040" color="#9b59b6"]
        T3  [label="get_account_history"          fillcolor="#251040" color="#9b59b6"]
        T4  [label="check_merchant_reputation"    fillcolor="#251040" color="#9b59b6"]
        T5  [label="get_transaction_velocity"     fillcolor="#251040" color="#9b59b6"]
        T6  [label="detect_geographic_anomaly"    fillcolor="#251040" color="#9b59b6"]
        T7  [label="search_fraud_patterns"        fillcolor="#251040" color="#9b59b6"]
        T8  [label="flag_transaction"             fillcolor="#300a0a" color="#e74c3c"]
        T9  [label="freeze_account"               fillcolor="#300a0a" color="#e74c3c"]
        T10 [label="get_recovery_recommendations" fillcolor="#251040" color="#9b59b6"]
    }

    subgraph cluster_memory {
        label="Agent Memory"
        style="filled,rounded"  fillcolor="#0a1a0a"  color="#27ae60"  fontcolor="white"
        MEM1 [label="Short-term: Scratchpad\n(in-context)"                  fillcolor="#0f2a15" color="#27ae60"]
        MEM2 [label="Long-term: FAISS\n8 fraud pattern types"               fillcolor="#0f2a15" color="#27ae60"]
        MEM3 [label="pending_actions[ ]\nFLAG / FREEZE  (status: PENDING)"  fillcolor="#2a1010" color="#e74c3c"]
    }

    subgraph cluster_data {
        label="Data and Models"
        style="filled,rounded"  fillcolor="#1a1000"  color="#f39c12"  fontcolor="white"
        DB1 [label="Transaction DB\n2,836 records"         shape=cylinder fillcolor="#2a1800" color="#f39c12"]
        DB2 [label="IsolationForest ML\n10 features"                      fillcolor="#2a1800" color="#f39c12"]
        DB3 [label="FAISS Index\nall-mpnet-base-v2"        shape=cylinder fillcolor="#2a1800" color="#f39c12"]
        DB4 [label="Merchant Risk DB\nBlacklist lookup"    shape=cylinder fillcolor="#2a1800" color="#f39c12"]
    }

    subgraph cluster_sqlite {
        label="SQLite Backend  (fraud_actions.db)"
        style="filled,rounded"  fillcolor="#0a0a20"  color="#3498db"  fontcolor="white"
        DB5 [label="actions\n(immutable audit trail)"    shape=cylinder fillcolor="#0f1535" color="#3498db"]
        DB6 [label="account_status\n(live freeze state)" shape=cylinder fillcolor="#0f1535" color="#3498db"]
    }

    USER   -> UI     [color="#4a9eff"]
    USER   -> ACTLOG [color="#4a9eff"]
    UI     -> LLM    [color="#4a9eff"]

    LOOP -> T1  [color="#9b59b6"]  LOOP -> T2  [color="#9b59b6"]  LOOP -> T3  [color="#9b59b6"]
    LOOP -> T4  [color="#9b59b6"]  LOOP -> T5  [color="#9b59b6"]  LOOP -> T6  [color="#9b59b6"]
    LOOP -> T7  [color="#9b59b6"]  LOOP -> T8  [color="#e74c3c"]  LOOP -> T9  [color="#e74c3c"]
    LOOP -> T10 [color="#9b59b6"]

    LLM -> MEM1 [color="#27ae60" style=dashed]
    LLM -> MEM2 [color="#27ae60" style=dashed]

    T8  -> MEM3 [color="#e74c3c" label="PENDING"]
    T9  -> MEM3 [color="#e74c3c" label="PENDING"]
    T9  -> DB6  [color="#3498db" style=dashed label="check if frozen"]
    T3  -> DB6  [color="#3498db" style=dashed label="read freeze status"]

    MEM3 -> HITL [color="#e74c3c" label="analyst approval required"]
    HITL -> DB5  [color="#3498db" label="log_action()"]
    HITL -> DB6  [color="#3498db" label="UPSERT on freeze"]

    ACTLOG -> DB5 [color="#3498db" style=dashed]
    ACTLOG -> DB6 [color="#3498db" style=dashed label="unfreeze()"]

    T1  -> DB1 [color="#f39c12" style=dashed]  T2  -> DB2 [color="#f39c12" style=dashed]
    T3  -> DB1 [color="#f39c12" style=dashed]  T4  -> DB4 [color="#f39c12" style=dashed]
    T5  -> DB1 [color="#f39c12" style=dashed]  T6  -> DB1 [color="#f39c12" style=dashed]
    T7  -> DB3 [color="#f39c12" style=dashed]  MEM2 -> DB3 [color="#27ae60" style=dashed]
}
"""
    st.graphviz_chart(dot_src, width='stretch')

    st.divider()
    st.subheader("Component Breakdown")

    col_b, col_t, col_m, col_h = st.columns(4)
    with col_b:
        st.markdown("#### 🧠 Brain (LLM)")
        st.markdown(
            "- **Model**: `nvidia/nemotron-3-super-120b-a12b:free`\n"
            "- **Access**: OpenRouter via `langchain-openrouter` (native reasoning support)\n"
            "- **Framework**: LangChain `create_agent`\n"
            "- **Pattern**: Tool-calling loop (system prompt + native function calling)\n"
            "- **Guardrails**: Structured output prompt, tool-grounded reasoning, "
            "controlled mock-only actions, per-tool input validation"
        )
    with col_t:
        st.markdown("#### 🔧 Tools (APIs / DBs)")
        st.markdown(
            "10 LangChain tools give the agent perception & action:\n\n"
            "- `get_transaction_details` — raw data fetch\n"
            "- `analyze_ml_risk_score` — IsolationForest score\n"
            "- `get_account_history` — 30-day pattern + freeze banner\n"
            "- `check_merchant_reputation` — blacklist lookup\n"
            "- `get_transaction_velocity` — burst detection\n"
            "- `detect_geographic_anomaly` — location check\n"
            "- `search_fraud_patterns` — FAISS semantic search\n"
            "- `flag_transaction` — queues FLAG (PENDING)\n"
            "- `freeze_account` — queues FREEZE (PENDING)\n"
            "- `get_recovery_recommendations` — customer guidance"
        )
    with col_m:
        st.markdown("#### 💾 Memory (State)")
        st.markdown(
            "**Short-term** (in-context)\n"
            "- Tool-calling scratchpad holds the current reasoning chain\n\n"
            "**Long-term** (vector)\n"
            "- FAISS IndexFlatIP over 8 fraud pattern documents\n"
            "- Embeddings: `all-mpnet-base-v2` (768-dim, local)\n"
            "- Semantic retrieval finds closest historical fraud type\n\n"
            "**pending_actions[ ]** (session)\n"
            "- Agent writes FLAG/FREEZE here with `status: PENDING`\n"
            "- No DB write until analyst approves via HITL panel"
        )
    with col_h:
        st.markdown("#### 🔐 HITL + Backend")
        st.markdown(
            "**Human-in-the-Loop Panel**\n"
            "- FRAUDULENT: Approve Freeze / Reject (flag only)\n"
            "- SUSPICIOUS: Escalate / Monitor / Downgrade\n"
            "- Two-phase commit: pending → approved → persisted\n\n"
            "**SQLite Backend** (`fraud_actions.db`)\n"
            "- `actions` table — immutable audit trail\n"
            "- `account_status` table — live freeze state\n"
            "- `get_account_history` reads freeze status\n"
            "- `freeze_account` short-circuits if already frozen\n"
            "- Actions Log tab: view, query, and unfreeze accounts"
        )
