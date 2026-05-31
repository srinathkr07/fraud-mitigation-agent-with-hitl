# 🛡️ FraudGuard AI — Intelligent Fraud Mitigation Agent

An AI-powered, self-contained fraud detection system that combines **traditional Machine Learning** (IsolationForest anomaly detection) with a **Generative AI agent** (powered by `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter) and a **Human-in-the-Loop (HITL)** approval layer to continuously monitor account activity, identify threats, and recommend or execute protective actions.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [High-Level Architecture](#high-level-architecture)
3. [Project Structure](#project-structure)
4. [Component Reference](#component-reference)
   - [app.py — Streamlit UI](#apppy--streamlit-ui)
   - [agent/fraud_agent.py — LLM Agent](#agentfraud_agentpy--llm-agent)
   - [agent/tools.py — Agent Tools](#agenttoolspy--agent-tools)
   - [agent/memory_store.py — Vector Memory](#agentmemory_storepy--vector-memory)
   - [ml/anomaly_detector.py — ML Model](#mlanomaly_detectorpy--ml-model)
   - [data/mock_data.py — Synthetic Data](#datamock_datapy--synthetic-data)
   - [backend/database.py — SQLite Persistence](#backenddatabasepy--sqlite-persistence)
5. [Data Flow: End-to-End Scenarios](#data-flow-end-to-end-scenarios)
6. [Human-in-the-Loop (HITL) Design](#human-in-the-loop-hitl-design)
7. [Risk Scoring System](#risk-scoring-system)
8. [Session State Reference](#session-state-reference)
9. [Database Schema](#database-schema)
10. [Configuration & Environment](#configuration--environment)
11. [Running the Application](#running-the-application)
12. [Embedded Fraud Scenarios](#embedded-fraud-scenarios)
13. [LLM Guardrails](#llm-guardrails)
14. [Known Limitations](#known-limitations)

---

## The Problem

Financial frauds costed the global economy over **$442 billion** in 2025. Traditional rule-based fraud systems suffer from three compounding weaknesses:

1. **High false-positive rates** — blunt threshold rules block legitimate transactions, frustrating customers.
2. **Slow human response loops** — by the time a fraud analyst reviews a flagged transaction, a velocity attacker may have swept an entire account.
3. **Binary detection only** — a transaction is identified as fraudulent or not, that's it.  


**FraudGuard AI** addresses this by combining:
- A traditional ML anomaly model (fast, deterministic)
- An LLM-based reasoning agent (nuanced, context-aware, actionable)
- A Human-in-the-Loop approval gate (prevents autonomous account freezing without analyst sign-off)
- A durable SQLite audit trail (every protective action is persisted across sessions)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI (app.py)                    │
│  📊 Dashboard · 🔍 Investigate · 🧠 Reasoning Log · 📋 Actions Log · 📐 Arch │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  user triggers investigation
┌──────────────────────────────▼──────────────────────────────────────┐
│                   FraudMitigationAgent  (fraud_agent.py)            │
│                                                                     │
│  LLM: nvidia/nemotron-3-super-120b-a12b:free (via OpenRouter)       │
│  Framework: LangChain create_agent  →  native tool-calling loop  │
└──────────┬──────────────────────────────────────┬───────────────────┘
           │ calls 10 tools                        │ returns messages list
┌──────────▼──────────────────────┐    ┌──────────▼───────────────────┐
│        Agent Tools  (tools.py)  │    │   Result dict                │
│                                 │    │   verdict / risk_score /     │
│  get_transaction_details        │    │   reasoning_steps /          │
│  analyze_ml_risk_score          │    │   final_answer               │
│  get_account_history ──► DB*    │    └──────────┬───────────────────┘
│  check_merchant_reputation      │               │
│  get_transaction_velocity       │    ┌──────────▼───────────────────┐
│  detect_geographic_anomaly      │    │      HITL Panel  (app.py)    │
│  search_fraud_patterns ──► FAISS│    │                              │
│  flag_transaction ──► pending[] │    │  FRAUDULENT: Approve/Reject  │
│  freeze_account ──► pending[]** │    │  SUSPICIOUS: Analyst review  │
│  get_recovery_recommendations   │    └──────────┬───────────────────┘
└──────────┬──────────────────────┘               │ analyst commits
           │ reads data                 ┌──────────▼───────────────────┐
┌──────────▼──────────────────────────┐│    backend/database.py        │
│   Data & ML Layer                   ││    SQLite: fraud_actions.db   │
│                                     ││    tables: actions            │
│  transactions_df — 2,836 rows       ││             account_status    │
│  ACCOUNTS dict   — 8 profiles       │└──────────────────────────────┘
│  FraudDetector   — IsolationForest  │
│  FraudVectorMemory — FAISS + SBERT  │
│  MERCHANTS / MERCHANT_BLACKLIST     │
└─────────────────────────────────────┘

* get_account_history queries account_status to prepend a live freeze banner
** freeze_account short-circuits if account_status.is_frozen = 1
```

---

## Project Structure

```
fraud-mitigation-agent/
├── app.py                         # Streamlit entry point — UI, HITL, session state
├── requirements.txt               # Python dependencies
├── sample_reasoning_log.json      # Pre-generated reference investigation trace
├── README.md                      # This document
├── .gitignore
│
├── pitch/
│   ├── slides.md                  # Hackathon pitch slides (in markdown format)
│   ├── slides.pdf                 # Hackathon pitch slides (in PDF format)
│
├── backend/
│   ├── __init__.py
│   ├── database.py                # SQLite persistence layer
│   └── fraud_actions.db           # Auto-created on first run (gitignored)
│
├── data/
│   ├── __init__.py
│   └── mock_data.py               # Synthetic data generator (seeded, deterministic)
│
├── ml/
│   ├── __init__.py
│   └── anomaly_detector.py        # IsolationForest anomaly detector + feature engineering
│
├── agent/
│   ├── __init__.py
│   ├── memory_store.py            # FAISS vector memory (sentence-transformers)
│   ├── tools.py                   # 10 LangChain tools
│   └── fraud_agent.py             # LangChain create_agent wrapper + result parser
│
└── logs/                          # JSON investigation logs (auto-created)
    └── YYYYMMDD_HHMMSS_<TXN>.json
```

---

## Component Reference

### `app.py` — Streamlit UI

The single-file entry point that wires together every other component.

**Startup sequence** (runs once per server start, results cached):
1. `init_db()` — creates SQLite tables if they don't exist
2. `load_data()` — generates/loads the mock transaction DataFrame, accounts dict, and fraud pattern corpus (cached with `@st.cache_resource`)
3. `load_detector(_df)` — trains the IsolationForest on the full transaction set (cached)
4. `load_memory(_patterns)` — builds the FAISS index over 8 fraud pattern documents (cached; downloads `all-mpnet-base-v2` ~420 MB on first run)
5. `enrich_df(_df, _detector)` — computes ML risk scores and anomaly flags for all 2,836 transactions (cached with `@st.cache_data`)

**Five tabs:**

| Tab | Content |
|-----|---------|
| 📊 Dashboard | 4 KPI metrics, top-50 transaction feed sorted by risk, risk distribution donut chart, top fraud accounts, timeline scatter chart |
| 🔍 Investigate | Transaction selector, single-click agent investigation, real-time reasoning trace, structured Final Report, HITL approval panels |
| 🧠 Reasoning Log | Dropdown over all saved `logs/*.json` files; renders full reasoning trace and structured report for any historical investigation |
| 📋 Actions Log | Live view of frozen accounts (with Unfreeze button), flagged transactions table, full audit log table — all read from SQLite |
| 📐 Architecture | Graphviz system diagram + component breakdown |

**Key functions:**

| Function | Purpose |
|----------|---------|
| `save_investigation_log(result, txn_df, accounts, committed_actions, hitl_decision)` | Serialises the full investigation result (including committed actions and HITL decision) to `logs/<timestamp>_<TXN>.json`; returns `(Path, investigation_id)` |
| `_parse_final_report(text)` | Extracts structured fields (verdict, risk score, fraud type, risk factors, actions, recovery steps) from the agent's markdown output using regex |
| `_render_final_report(text)` | Renders the parsed report with Streamlit native elements (error/warning/success banners, metric cards, bullet lists) |
| `render_log(log_data)` | Renders a saved JSON log — used in both the Reasoning Log tab and the inline investigation view |
| `_build_log_trace(reasoning_steps)` | Converts agent `reasoning_steps` list to the sample-log-compatible `reasoning_trace` format |

**Session state keys** (see full table in [Session State Reference](#session-state-reference)).

---

### `agent/fraud_agent.py` — LLM Agent

Wraps a LangChain `create_agent` (with native tool-calling) in a single `FraudMitigationAgent` class.

**Model:** `nvidia/nemotron-3-super-120b-a12b:free`
- 120B-parameter hybrid Mamba-Transformer Mixture-of-Experts (12B active parameters)
- 262K token context window
- Native tool-calling (function calling)
- Native reasoning blocks (`content_blocks` with `type: "reasoning"`)
- Accessed via `langchain-openrouter` with `reasoning={"effort": "medium", "summary": "auto"}`

**Investigation protocol (system prompt):**
The LLM is instructed to always follow 10 steps in order:

```
1. get_transaction_details       — always first
2. analyze_ml_risk_score         — ML anomaly score
3. get_account_history           — 30-day spending pattern
4. check_merchant_reputation     — blacklist check
5. get_transaction_velocity      — burst/frequency detection
6. detect_geographic_anomaly     — impossible travel check
7. search_fraud_patterns         — FAISS semantic search
8. flag_transaction              — queue FLAG for analyst review (PENDING)
9. freeze_account                — queue FREEZE for analyst review (PENDING)
10. get_recovery_recommendations — always retrieve customer guidance
```

**Verdict scale:**
- `LEGITIMATE` — risk score < 35, no major anomaly flags
- `SUSPICIOUS` — risk score 35–69, warrants monitoring
- `FRAUDULENT` — risk score ≥ 70 OR critical anomaly flags

**`investigate(transaction_id)` return dict:**

| Key | Type | Description |
|-----|------|-------------|
| `transaction_id` | str | The investigated transaction |
| `final_answer` | str | The structured markdown report |
| `reasoning_steps` | list[dict] | One entry per tool call: `thought`, `reasoning`, `action`, `action_input`, `observation` |
| `verdict` | str | `LEGITIMATE` \| `SUSPICIOUS` \| `FRAUDULENT` \| `UNKNOWN` |
| `risk_score` | int \| None | Parsed from report (0–100) |
| `elapsed_seconds` | float | Wall-clock duration of the investigation |
| `error` | str \| None | Human-readable error message if the API call failed |
| `error_type` | str \| None | `auth_error` \| `rate_limit` \| `quota_exhausted` \| `connection_error` \| `unknown_error` |

**`_extract_verdict` algorithm:**
1. `re.search` (unanchored) for `**VERDICT**: <word>` with optional markdown bold markers anywhere in the text
2. Fallback: scans for FRAUDULENT first (skipping if "not fraudulent" precedes it), then SUSPICIOUS, then LEGITIMATE — descending severity order
3. Returns `"UNKNOWN"` if nothing found

---

### `agent/tools.py` — Agent Tools

All 10 tools are created by the **factory function** `create_fraud_tools(...)`, which closes over injected dependencies — there is no global state.

```python
def create_fraud_tools(
    transactions_df: pd.DataFrame,
    accounts: dict,
    detector: FraudDetector,
    memory: FraudVectorMemory,
    pending_actions: list,          # mutable, shared with session_state
) -> list
```

The `pending_actions` list is a direct reference to `st.session_state.pending_actions`. When `flag_transaction` or `freeze_account` append to it, those changes are immediately visible in session state without any return value — this is the key mechanism of the HITL two-phase commit.

**Tool summary:**

| # | Tool | Input | Key behaviour |
|---|------|-------|---------------|
| 1 | `get_transaction_details` | `transaction_id: str` | Returns full transaction row + account metadata |
| 2 | `analyze_ml_risk_score` | `transaction_id: str` | Calls `detector.get_risk_assessment()`; returns score 0–100, level, flags, explanation |
| 3 | `get_account_history` | `account_id: str` | Queries SQLite for live freeze status (prepends banner if frozen); returns 30-day summary + 8 most recent transactions |
| 4 | `check_merchant_reputation` | `merchant_name: str` | Fuzzy + partial name match against MERCHANTS dict; returns risk level and blacklist status |
| 5 | `get_transaction_velocity` | `transaction_id: str` | Counts transactions in the 1h and 24h windows before the given transaction; alerts if > 3 in 1h |
| 6 | `detect_geographic_anomaly` | `transaction_id: str` | Compares transaction location against `usual_locations` + historical locations; detects impossible travel (< 2h gap, international) |
| 7 | `search_fraud_patterns` | `description: str` | Embeds the description, searches FAISS index, returns top-3 matching fraud pattern types with similarity score |
| 8 | `flag_transaction` | `"<txn_id> \| <reason>"` | Appends `{action: FLAG_TRANSACTION, status: PENDING, ...}` to `pending_actions`; returns "queued — pending analyst approval" |
| 9 | `freeze_account` | `"<acc_id> \| <reason>"` | Checks SQLite for existing freeze; if already frozen returns early with warning; otherwise appends `{action: FREEZE_ACCOUNT, status: PENDING, ...}` to `pending_actions` |
| 10 | `get_recovery_recommendations` | `fraud_type: str` | Looks up a hardcoded recovery steps dict keyed by fraud type; falls back to `"unknown"` key |

**Recovery recommendation keys:** `card_testing`, `card_testing_followup`, `geo_anomaly`, `velocity_attack`, `high_value_night`, `suspicious_merchant`, `account_takeover`, `unknown`

---

### `agent/memory_store.py` — Vector Memory

Provides long-term semantic memory over the 8 known fraud pattern documents.

**Model:** `all-mpnet-base-v2` (sentence-transformers, 768-dimensional embeddings, runs locally, ~420 MB download on first use)

**Index:** `faiss.IndexFlatIP` — exact brute-force inner product search over L2-normalised vectors (equivalent to cosine similarity)

**`build_index(patterns)`** — encodes all patterns as a flat string (`name + description + indicators + severity`) and adds to the FAISS index. Called once at startup via `@st.cache_resource`.

**`search(query, k=3)`** — encodes the query, L2-normalises, searches the index; returns top-k results each augmented with a `similarity_score` (0–1 range after normalisation).

**Pattern-to-text serialisation:**
```
{name}: {description} Indicators: {indicator1}; {indicator2}. Severity: {severity}.
```

---

### `ml/anomaly_detector.py` — ML Model

**Algorithm:** `sklearn.ensemble.IsolationForest`
- `n_estimators=200`, `contamination=0.10`, `random_state=42`
- **Trained on normal transactions only** — fraud rows are excluded from fitting to give a clean decision boundary

**10 Features:**

| # | Feature | Description |
|---|---------|-------------|
| 0 | `log_amount` | `log1p(amount)` — reduces skew from large transactions |
| 1 | `hour_of_day` | 0–23 |
| 2 | `day_of_week` | 0–6 (Monday=0) |
| 3 | `is_weekend` | 1 if day_of_week ≥ 5 |
| 4 | `is_nighttime` | 1 if hour < 6 or hour ≥ 22 |
| 5 | `is_international` | From transaction flag |
| 6 | `merchant_risk_score` | low=0, medium=1, high=2 |
| 7 | `velocity_1h` | Count of prior account transactions within 1 hour |
| 8 | `velocity_24h` | Count of prior account transactions within 24 hours |
| 9 | `amount_zscore` | (amount − expanding mean) / expanding std, clipped at 10 |

Features are scaled with `StandardScaler` before fitting.

**Score normalisation:** IsolationForest `decision_function` returns higher values for normal observations. To get an intuitive risk score:
```
risk_score = clip((max_score − raw_score) / (max_score − min_score) * 100, 0, 100)
```

**Risk levels:**
- `LOW` — score 0–34
- `MEDIUM` — score 35–69
- `HIGH` — score 70–79
- `CRITICAL` — score 80–100

**Anomaly flag rules** (applied on raw feature vector independently of the ML score):

| Flag | Condition |
|------|-----------|
| `HIGH_AMOUNT` | feature[0] (log_amount) > log1p(500) ≈ 6.22 |
| `NIGHTTIME_TRANSACTION` | feature[4] > 0.5 (is_nighttime=1) |
| `INTERNATIONAL_TRANSACTION` | feature[5] > 0.5 |
| `HIGH_RISK_MERCHANT` | feature[6] > 1.5 (merchant_risk_score=2, i.e. "high") |
| `HIGH_VELOCITY_1H` | feature[7] > 3 (more than 3 prior transactions in last hour) |
| `HIGH_VELOCITY_24H` | feature[8] > 15 |
| `UNUSUAL_AMOUNT_FOR_ACCOUNT` | feature[9] > 2.5 (amount > 2.5σ above account's expanding mean) |

---

### `data/mock_data.py` — Synthetic Data

Generates a fully synthetic, reproducible dataset. `random.seed(42)` and `np.random.seed(42)` ensure identical output on every run.

**Dataset:** 2,836 transactions spanning 2024-03-01 to 2024-06-01 across 8 accounts.

**Account profiles (8 accounts):**

| ID | Name | Avg Spend | Daily Limit | Usual Locations |
|----|------|-----------|-------------|-----------------|
| ACC001 | Alice Johnson | $75 | $2,000 | New York, NY; Newark, NJ |
| ACC002 | Bob Martinez | $120 | $3,000 | Chicago, IL; Evanston, IL |
| ACC003 | Carol Williams | $95 | $2,500 | Los Angeles, CA; Santa Monica, CA |
| ACC004 | David Chen | $200 | $5,000 | Seattle, WA; Bellevue, WA |
| ACC005 | Emma Davis | $55 | $1,500 | Miami, FL; Fort Lauderdale, FL |
| ACC006 | Frank Wilson | $150 | $4,000 | Boston, MA; Cambridge, MA |
| ACC007 | Grace Lee | $180 | $4,500 | San Francisco, CA; Oakland, CA |
| ACC008 | Henry Brown | $110 | $3,000 | Austin, TX; Round Rock, TX |

**Transaction schema:**

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | str | UUID-based, prefix "TXN" |
| `account_id` | str | "ACC001"–"ACC008" |
| `timestamp` | datetime | Transaction datetime |
| `amount` | float | Transaction amount ($) |
| `merchant` | str | Merchant name |
| `category` | str | Spending category |
| `location` | str | City/region string |
| `is_international` | bool | True if flagged as international |
| `merchant_risk_level` | str | "low" / "medium" / "high" |
| `is_fraud` | bool | Ground-truth fraud label (used to train ML on normal rows only) |
| `fraud_type` | str \| None | Fraud pattern key (e.g. `"card_testing"`, `"geo_anomaly"`) or None for normal transactions |

**KNOWN_FRAUD_PATTERNS (8 patterns):** Card Testing Attack, Card Skimming / POS Compromise, Account Takeover Fraud, Geographic Anomaly / International Fraud, High-Value Nighttime Transaction, Velocity Attack, Suspicious / Ghost Merchant, Friendly Fraud / Chargeback Fraud — each with `name`, `description`, `indicators[]`, `severity`, and `mitigation` fields.

---

### `backend/database.py` — SQLite Persistence

Provides a durable audit trail and live account-freeze state. The database file is at `backend/fraud_actions.db` and is created automatically by `init_db()` on every application start.

**Connection model:** Every function opens a fresh `sqlite3.Connection` and closes it in a `try/finally` block. `check_same_thread=False` allows Streamlit's multi-thread execution model to work safely.

**Public API:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `init_db()` | `() → None` | `CREATE TABLE IF NOT EXISTS` for both tables. Safe to call on every startup. |
| `log_action(entry, analyst_decision, analyst_notes, risk_score, agent_verdict, investigation_id)` | `→ str` | INSERT into `actions`. If `action_type == "FREEZE_ACCOUNT"`, also UPSERTs `account_status` (sets `is_frozen=1`). Returns `action_id` (UUID4). |
| `unfreeze_account(account_id)` | `→ None` | Reads current freeze state; if frozen, INSERTs an `UNFREEZE_ACCOUNT` audit entry then sets `is_frozen=0` in `account_status`. No-op if account is not frozen (prevents phantom audit entries). |
| `get_account_freeze_status(account_id)` | `→ dict \| None` | Returns current `account_status` row or None. Used by `get_account_history` and `freeze_account` tools. |
| `get_all_actions(limit=200)` | `→ list[dict]` | All actions newest-first. |
| `get_frozen_accounts()` | `→ list[dict]` | All rows with `is_frozen=1`. |
| `get_flagged_transactions()` | `→ list[dict]` | All `FLAG_TRANSACTION` rows, newest-first. |

---

## Data Flow: End-to-End Scenarios

### Scenario 1 — FRAUDULENT verdict, analyst approves freeze

```
1. User clicks "Run Agent Investigation" for a FRAUDULENT transaction
2. app.py clears session state: action_log=[], pending_actions=[], hitl_pending=False
3. create_fraud_tools() wires tools to st.session_state.pending_actions (by reference)
4. FraudMitigationAgent.investigate() → LangChain invokes 10 tools in protocol order
5. flag_transaction tool → pending_actions.append({PENDING, FLAG_TRANSACTION, ...})
6. freeze_account tool   → checks DB (not frozen) → pending_actions.append({PENDING, FREEZE_ACCOUNT, ...})
7. Agent returns result dict: verdict=FRAUDULENT, risk_score=100, reasoning_steps, final_answer
8. app.py detects: verdict=FRAUDULENT AND freeze queued → hitl_pending=True
9. UI renders HITL panel: "Approve — Execute Freeze" / "Reject Freeze — Flag Only"
10a. Analyst clicks APPROVE:
    → all pending_actions moved to action_log (status key stripped)
    → pending_actions cleared, hitl_pending=False
    → hitl_decision = {decision: "approved", timestamp: ...}
    → save_investigation_log(...) → logs/YYYYMMDD_HHMMSS_TXN.json + returns investigation_id
    → log_action(FLAG_TRANSACTION, analyst_decision="approved", ...) → INSERT into actions
    → log_action(FREEZE_ACCOUNT, analyst_decision="approved", ...) → INSERT into actions + UPSERT account_status (is_frozen=1)
    → st.rerun() → HITL panel replaced by green/red decision banner + Protective Actions expander
```

### Scenario 2 — FRAUDULENT verdict, analyst rejects freeze

```
Steps 1–9 identical to Scenario 1.
10b. Analyst clicks REJECT FREEZE:
    → only FLAG_TRANSACTION items moved to action_log (FREEZE_ACCOUNT items dropped)
    → pending_actions cleared, hitl_pending=False
    → hitl_decision = {decision: "rejected_freeze", timestamp: ...}
    → save_investigation_log(...)
    → log_action(FLAG_TRANSACTION, analyst_decision="rejected_freeze", ...) → INSERT into actions
    → st.rerun()
```

### Scenario 3 — SUSPICIOUS verdict, analyst submits determination

```
Steps 1–7 similar; step 8: verdict=SUSPICIOUS → hitl_pending=True
9. UI renders Analyst Review panel: radio (escalate / monitor / downgrade) + notes textarea
10. Analyst selects verdict and clicks "Submit Decision":
    - "escalate":
        → action_log += FLAG_TRANSACTION (analyst-generated)
        → action_log += FREEZE_ACCOUNT (analyst-generated)
        → log_action × 2 (analyst_decision="escalate")
    - "monitor":
        → action_log += FLAG_TRANSACTION only
        → log_action × 1 (analyst_decision="monitor")
    - "downgrade":
        → action_log remains empty
        → log_action not called (no actions to record)
    → pending_actions cleared (agent's original pending actions dropped)
    → hitl_decision = {decision: ..., notes: ..., timestamp: ...}
    → save_investigation_log(...)
    → st.rerun()
```

### Scenario 4 — LEGITIMATE / no freeze queued (auto-commit)

```
Steps 1–7 similar; step 8: verdict=LEGITIMATE OR (FRAUDULENT with no FREEZE queued)
9. Auto-commit path:
    → all pending_actions (may contain FLAG only) moved to action_log
    → pending_actions cleared, hitl_pending remains False
    → save_investigation_log(... hitl_decision=None)
    → log_action(analyst_decision="auto") for each committed action
    → toast notification, no HITL panel shown
```

---

## Human-in-the-Loop (HITL) Design

### Why HITL?

Account freezes are **irreversible** in most banking systems and cause immediate customer disruption. Delegating this action to an LLM without human review creates unacceptable risk — both from false positives (legitimate customers locked out) and from model errors. HITL ensures that the agent can *recommend* a freeze instantly, but a human must *approve* it before it takes effect.

### Two-Phase Commit

The design separates agent action from analyst approval:

**Phase 1 — Agent queues actions:**
- `flag_transaction` and `freeze_account` tools write to `pending_actions` with `status: "PENDING"`
- No database writes occur
- No session state changes beyond appending to `pending_actions`

**Phase 2 — Analyst commits actions:**
- Analyst reviews the full investigation evidence, then clicks a decision button
- `pending_actions` items are moved to `action_log` (status key stripped)
- `save_investigation_log()` is called — JSON persisted to `logs/`
- `log_action()` is called for each committed action — SQLite persisted
- `st.rerun()` refreshes the UI

This guarantees: **no action ever reaches the database before a human has reviewed it**.

### HITL Touchpoints

| Trigger | Panel type | Options |
|---------|-----------|---------|
| Verdict = `FRAUDULENT` AND `FREEZE_ACCOUNT` queued | Freeze confirmation | Approve (execute freeze + flag) / Reject (flag only) |
| Verdict = `SUSPICIOUS` | Analyst review | Escalate to FRAUDULENT / Monitor only / Downgrade to LEGITIMATE |
| Verdict = `FRAUDULENT`, no freeze queued (flag only) | Auto-commit | No panel shown; FLAG auto-committed with `analyst_decision="auto"` |
| Verdict = `LEGITIMATE` | Auto-commit | No panel shown; any minor actions auto-committed with `analyst_decision="auto"` |

### Stale-HITL Warning

If the user switches the transaction selector while a HITL decision is pending for a different transaction, a warning banner appears:
```
"A HITL decision is still pending for TXN<X>. Re-select that transaction to complete it,
or run a new investigation to discard it."
```
Starting a new investigation always resets `hitl_pending=False` and clears `pending_actions`.

---

## Risk Scoring System

Three independent layers each contribute to the final risk picture:

### Layer 1 — IsolationForest ML Score (0–100)

Computed at startup for all 2,836 transactions and cached in memory. Uses 10 engineered features including log-transformed amount, velocity counts, geographic signals, and per-account z-scores. Trained exclusively on normal (non-fraud) transactions for a clean decision boundary.

### Layer 2 — Rule-Based Anomaly Flags

Seven binary flag rules applied independently on the raw feature vector. These fire regardless of the ML score and are surfaced to both the dashboard and the agent tools. Flags are additive — a transaction can trigger multiple flags simultaneously.

### Layer 3 — LLM Verdict (LEGITIMATE / SUSPICIOUS / FRAUDULENT)

The agent correlates the ML score, anomaly flags, merchant reputation, account history, velocity patterns, geographic signals, and FAISS fraud pattern matches to produce a final verdict. The verdict is deterministic given the tool outputs (not generated from LLM "intuition") because the system prompt enforces tool-grounded reasoning.

**Verdict threshold guidance (from system prompt):**
- `LEGITIMATE` if risk < 35 and no critical flags
- `SUSPICIOUS` if risk 35–69
- `FRAUDULENT` if risk ≥ 70 OR critical anomaly flags present

---

## Session State Reference

All Streamlit session state keys and their lifecycle:

| Key | Type | Initial value | Description |
|-----|------|---------------|-------------|
| `action_log` | `list[dict]` | `[]` | Committed protective actions for the current investigation. Cleared on each new investigation start. |
| `investigation` | `dict \| None` | `None` | The full `result` dict from the most recent `FraudMitigationAgent.investigate()` call. |
| `api_key` | `str` | From env | OpenRouter API key. Persists across reruns. |
| `show_sample_log` | `bool` | `False` | Whether to show the sample log after an API error. |
| `pending_actions` | `list[dict]` | `[]` | Agent-queued actions waiting for human approval. Items have `status: "PENDING"`. Cleared on new investigation and on HITL commit. Passed by reference into the tool factory. |
| `hitl_pending` | `bool` | `False` | True when FRAUDULENT+freeze or SUSPICIOUS verdict received and analyst hasn't yet submitted a decision. |
| `hitl_decision` | `dict \| None` | `None` | The analyst's committed decision: `{decision, timestamp}` or `{decision, notes, timestamp}`. Displayed as a banner after commit. |

---

## Database Schema

### Table: `actions`

The complete immutable audit trail. One row per protective action.

| Column | Type | Description |
|--------|------|-------------|
| `action_id` | TEXT PRIMARY KEY | UUID4 generated at insert time |
| `action_type` | TEXT NOT NULL | `FLAG_TRANSACTION` \| `FREEZE_ACCOUNT` \| `UNFREEZE_ACCOUNT` |
| `transaction_id` | TEXT | Transaction ID. NULL for FREEZE_ACCOUNT and UNFREEZE_ACCOUNT rows. Also NULL for analyst-escalated freeze rows from the SUSPICIOUS path (the freeze is tied to an investigation but not a specific transaction ID). |
| `account_id` | TEXT NOT NULL | Account ID |
| `account_holder` | TEXT | Account holder name |
| `reason` | TEXT | Human-readable reason for the action |
| `risk_score` | INTEGER | ML risk score at time of investigation (0–100) |
| `agent_verdict` | TEXT | `FRAUDULENT` \| `SUSPICIOUS` \| `LEGITIMATE` \| `UNKNOWN` |
| `analyst_decision` | TEXT | `auto` \| `approved` \| `rejected_freeze` \| `escalate` \| `monitor` \| `downgrade` \| `analyst` (unfreeze) |
| `analyst_notes` | TEXT | Free-text notes from the analyst (may be NULL) |
| `investigation_id` | TEXT | Links to the JSON log file stem (e.g. `INV-20260508-143022-042`) |
| `created_at` | TEXT NOT NULL | ISO 8601 UTC timestamp |

### Table: `account_status`

Live mutable state. One row per account that has ever been frozen. `is_frozen=0` rows remain as history.

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | TEXT PRIMARY KEY | Account ID |
| `holder_name` | TEXT | Account holder name |
| `is_frozen` | INTEGER NOT NULL DEFAULT 0 | 1 if currently frozen, 0 if unfrozen |
| `freeze_reason` | TEXT | Reason from the most recent freeze action (NULL after unfreeze) |
| `freeze_action_id` | TEXT | FK → `actions.action_id` of the freeze row (NULL after unfreeze) |
| `frozen_at` | TEXT | UTC timestamp of most recent freeze (NULL after unfreeze) |
| `last_updated` | TEXT NOT NULL | UTC timestamp of last write to this row |

**Investigation ID format:** `INV-YYYYMMDD-HHMMSS-mmm` (millisecond precision to prevent collision between investigations started in the same second)

---

## Configuration & Environment

```bash
# .env file (project root)
OPENROUTER_API_KEY=sk-or-v1-your_key_here
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes (for live investigations) | — | Free key at openrouter.ai/keys |
| `OMP_NUM_THREADS` | No | `1` (set by app.py) | Prevents numpy/torch from spawning excess threads |
| `MKL_NUM_THREADS` | No | `1` (set by app.py) | Same as above for MKL |
| `TOKENIZERS_PARALLELISM` | No | `false` (set by app.py) | Prevents HuggingFace tokenizer semaphore warnings |

The API key can also be pasted directly into the Investigate tab UI. The `.env` file takes precedence — if a valid key is found there, the UI input field is hidden.

---

## Running the Application

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd fraud-mitigation-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **First run note:** Building the FAISS index downloads `all-mpnet-base-v2` (~420 MB). This happens once and is cached by sentence-transformers.

### API key

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-your_key_here' > .env
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys). The `nvidia/nemotron-3-super-120b-a12b:free` model is free-tier but has daily quota limits.

### Start the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### First investigation

1. Go to **🔍 Investigate**
2. Sort by "Risk Score ↓" — top entries are the embedded fraud scenarios
3. Select a transaction with risk score ≥ 70
4. Click **🚀 Run Agent Investigation** (takes 30–90 seconds depending on LLM latency)
5. Review the reasoning trace and structured report
6. If FRAUDULENT: use the HITL panel to approve or reject the freeze
7. If SUSPICIOUS: submit your analyst determination
8. Visit **📋 Actions Log** to see the persisted audit trail

### Verifying the database

```bash
sqlite3 backend/fraud_actions.db "SELECT action_type, account_id, analyst_decision, created_at FROM actions ORDER BY created_at DESC LIMIT 10;"
sqlite3 backend/fraud_actions.db "SELECT account_id, holder_name, is_frozen, frozen_at FROM account_status;"
```

---

## Embedded Fraud Scenarios

Five distinct fraud patterns are baked into the dataset for demonstration:

| # | Account | Pattern | Key Signals |
|---|---------|---------|-------------|
| 1 | ACC003 – Carol Williams | **Card Testing Attack** | 12 micro-charges ($0.50–$4.99) followed by $899.99 to a blacklisted offshore merchant |
| 2 | ACC001 – Alice Johnson | **High-Value Night Transaction** | $4,500 at 2:47 AM, blacklisted merchant, international flag |
| 3 | ACC002 – Bob Martinez | **Geographic Anomaly** | Transactions from Lagos (Nigeria) and Moscow (Russia) against Chicago baseline |
| 4 | ACC004 – David Chen | **Velocity Attack** | 18 transactions in 25.5 minutes across 5 high-risk merchants (3 blacklisted) via VPN |
| 5 | ACC007 – Grace Lee | **Suspicious Merchant** | $1,200 + $750 at QuickLoans.net and CryptoExchange Pro at midnight |

---

## LLM Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| **Structured output enforcement** | System prompt requires the final answer to follow a rigid template (VERDICT / Risk Score / Fraud Type / Key Risk Factors / Actions Taken / Recovery Steps). Sections cannot be omitted. |
| **Tool-grounded reasoning** | The agent must call tools to retrieve facts — it cannot invent transaction details, scores, or account history. All data comes from deterministic tool outputs. |
| **Pending-only actions** | `flag_transaction` and `freeze_account` tools write only to `pending_actions` (in-memory). No database write occurs until a human clicks a commit button. |
| **Duplicate freeze prevention** | `freeze_account` checks `account_status` before queuing. Already-frozen accounts return an immediate short-circuit message, preventing duplicate entries. |
| **Controlled loop depth** | LangChain's `create_agent()` bounds the tool-calling iteration. The system prompt lists exactly 10 steps, making the expected call count deterministic. |
| **Parse error recovery** | `_render_final_report` falls back to raw markdown if structured fields cannot be parsed — no crash on unexpected model output. |
| **Error classification** | HTTP error codes from OpenRouter are mapped to user-facing messages (`auth_error`, `rate_limit`, `quota_exhausted`, `connection_error`) rather than raw exception text. |
| **Verdict false-positive prevention** | `_extract_verdict` uses a regex anchored to the `VERDICT:` line before falling back to substring search, preventing "NOT FRAUDULENT" matching as FRAUDULENT. |
| **Input sanitisation** | Tool input parsers validate format before processing; malformed inputs return controlled error strings, not unhandled exceptions. |

---

## Known Limitations

1. **All data is synthetic** — the transaction dataset, account profiles, and merchant database are fully mock. No real financial data is processed.

2. **Free-tier LLM quota** — `nvidia/nemotron-3-super-120b-a12b:free` has daily limits on OpenRouter. Investigations may fail with `quota_exhausted` after heavy use. A pre-recorded sample log is shown as fallback.

3. **Session-scoped action log** — `st.session_state.action_log` is cleared on each new investigation. The database preserves history across sessions, but the in-session "Protective Actions" expander only shows the current investigation's committed actions.

4. **Single-user design** — the application runs as a single Streamlit process. Concurrent multi-user access will share a single SQLite file (safe for reads; writes are serialised by SQLite's WAL mode, but session state is not shared between browser tabs).

5. **IsolationForest trained at startup** — new transactions added to the dataset after startup will not receive ML scores unless the server is restarted. The `@st.cache_resource` decorator persists the trained model for the server lifetime.

6. **No real protective actions** — `flag_transaction` and `freeze_account` write only to the local SQLite database and `st.session_state`. There is no integration with any real banking or transaction processing system.
