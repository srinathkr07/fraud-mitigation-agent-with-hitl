---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-size: 26px;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #34495e;
    font-size: 32px;
  }
  h3 {
    color: #7f8c8d;
    font-size: 28px;
  }
  table {
    font-size: 20px;
  }
  pre {
    font-size: 16px;
  }
  blockquote {
    background: #f0f0f0;
    border-left: 5px solid #e74c3c;
    padding: 10px 20px;
    font-style: italic;
  }
  ul, ol {
    font-size: 24px;
  }
  .small-text {
    font-size: 22px;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# **🛡️ _FraudGuard AI_**

## Intelligent Fraud Detection with Human Oversight

**GrabHack 2.0 | May 2026**

by Team Runtime Terror

---

<!-- _class: lead -->

# **⚠️ The Current Problem**

---

## ⚡ Financial Fraud Is Getting Faster Than Human Response

> **$442 billion** lost globally to financial frauds in 2025 ([Interpol](https://m.economictimes.com/news/international/world-news/ai-scams-propel-global-financial-fraud-to-442-billion-in-2025-warns-interpol/articleshow/129612343.cms))

Traditional rule-based fraud systems have three compounding failures:

<div align="center">

| Pain Point | What Breaks |
|---|---|
| **High false-positive rates** | Blunt threshold rules block real customers |
| **Slow human review loops** | Velocity attackers drain accounts before review |
| **Binary detection only** | Can't identify *what kind* of fraud |

</div>

---

## 🎯 The Gap No One Is Closing

Existing systems are either:

- **ML-only** — fast, but can't reason or explain decisions
- **Rule-based** — transparent, but brittle and easy to evade
- **Human-only review** — accurate, but too slow at scale

**No system combines statistical speed, contextual LLM reasoning, and safe human oversight — until now.**

---

<!-- _class: lead -->

# **Our Solution: 🛡️ _FraudGuard AI_**

---

## Introducing 🛡️ _FraudGuard AI_

> An AI-powered fraud investigation agent that **detects**, **classifies**, and **acts** — in minutes, with a human safety gate before any irreversible action.

**Investigation in three layers:**

<div class="small-text">

```
Transaction flagged by ML (< 1 second)
             ↓
  [ IsolationForest — 0–100 risk score ]
             ↓
  [ LLM Agent — 10 tools, full reasoning trace ]
     Classifies fraud type · Correlates signals
             ↓
  [ HITL Panel — Analyst approves/rejects ]
     Flag transaction · Freeze account · Provide Recovery steps
             ↓
  [ SQLite Audit Trail — Every action persisted ]
```

</div>

---

## Three Key Differentiators

1. **Classifies** fraud type — not just a binary flag
   - Examples: _velocity attack_, _geographic anomaly_, _card testing_

2. **Reasons** across 10 signals simultaneously in one trace
   - Amount, location, merchant, velocity, history, patterns

3. **Human-in-the-loop safety gate**
   - The agent recommends; the analyst approves before any account is frozen

---

<!-- _class: lead -->

# **🏗️ System Architecture**

---

## How It Works End-to-End

<div class="small-text">

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI (app.py)                    │
│  📊 Dashboard · 🔍 Investigate · 🧠 Reasoning Log · 📋 Actions Log  │
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

</div>

---

## 🛠️ Tech Stack

<div class="small-text">

| Layer | Technology | Why |
|---|---|---|
| **LLM** | `nvidia/nemotron-3-super-120b-a12b:free` | 120B MoE, 262K context — **free** via _OpenRouter_ |
| **Agent** | LangChain `create_agent` | Native tool-calling loop |
| **ML** | scikit-learn `IsolationForest` | Unsupervised anomaly detection |
| **Vector Store** | `faiss-cpu` + `sentence-transformers` | Local semantic search |
| **Embeddings** | `all-mpnet-base-v2` (420MB) | Privacy-preserving |
| **DB** | SQLite | Durable audit trail |
| **UI** | Streamlit + Plotly | Dashboard + reasoning viewer |

</div>

---

<!-- _class: lead -->

# **🧠 The Intelligence Stack**

---

## Three Layers of Intelligence

### 🤖 Layer 1 — IsolationForest ML
- Trained at startup on **normal** transactions only
- Scores every transaction 0–100 in **< 1 second**
- **10 features:** log-amount, hour, day, velocity, international flag, merchant risk, z-score
- **7 anomaly flags:** HIGH_AMOUNT, NIGHTTIME, INTERNATIONAL, HIGH_RISK_MERCHANT, HIGH_VELOCITY_1H, HIGH_VELOCITY_24H, UNUSUAL_AMOUNT

---

## Three Layers (continued)

### 💡 Layer 2 — LLM Reasoning Agent
- Receives ML score + 7 flags, calls **10 specialist tools**
- Classifies *type* of fraud via FAISS semantic search
- Verdict: **LEGITIMATE** (< 35) · **SUSPICIOUS** (35–69) · **FRAUDULENT** (≥ 70)
- Full reasoning trace logged

### 👤 Layer 3 — Human-in-the-Loop Gate
- Agent queues actions as **PENDING** — no DB write until approval
- Two HITL touchpoints: FRAUDULENT freeze confirm · SUSPICIOUS analyst review
- Every committed action logged to SQLite with decision, notes, timestamp

---

<!-- _class: lead -->

# **🤖🤝👤 Human-in-the-Loop Design**

---

## Why HITL? Because Account Freezes Are Irreversible

Freezing a legitimate customer's account causes:
- Immediate customer distress and support escalation
- Regulatory liability if done without proper review
- Trust damage that's hard to recover

---

## Two-Phase Commit Model

**Phase 1 — Agent queues actions (PENDING, no DB write)**
- flag_transaction → pending_actions[]
- freeze_account → pending_actions[]

**Phase 2 — Analyst commits (DB write only on approval)**

| Verdict | Actions |
|---|---|
| **FRAUDULENT** | ✅ Approve → FLAG + FREEZE committed<br>❌ Reject → FLAG only |
| **SUSPICIOUS** | 🔴 Escalate → FLAG + FREEZE<br>🟡 Monitor → FLAG only<br>🟢 Downgrade → No action |

---

## Result

The agent is fast and thorough; the human is the final safety gate.

---

<!-- _class: lead -->

# **🎬 Live Demo**

---

## 🔍 What You'll See — Dashboard & Investigate

**📊 Dashboard**
- 2,836 transactions with ML risk scores
- 4 KPIs: Total · Flagged · High-Risk · Amount at Risk
- Risk distribution chart · Top fraud accounts · Timeline scatter

**🔍 Investigate**
1. Select a high-risk transaction (sort by Risk Score ↓)
2. Click **Run Agent Investigation** — watch 10 tools fire (~3-5 minutes)
3. Read full reasoning trace: tool inputs, outputs, LLM reasoning
4. See Final Report: VERDICT · Risk Score · Fraud Type · Recovery Steps
5. Use **HITL Panel** to approve/reject/escalate

---

## 📄 What You'll See — Logs & Architecture

**🧠 Reasoning Log**
- Browse all saved investigation logs
- Complete reasoning trace replay for any past investigation

**📋 Actions Log**
- Live frozen accounts list with one-click Unfreeze
- All flagged transactions with analyst decisions
- Full audit trail: every FLAG, FREEZE, UNFREEZE with timestamp and notes

**🔧 Architecture**
- System diagram and component breakdown

---

## Suggested Demo Transaction 🎲

**Sort by Risk Score ↓, pick `ACC003` (Carol Williams)**
- Triggers **Card Testing Attack** pattern
- Risk Score: **100/100**

---

<!-- _class: lead -->

# **🎯 Embedded Fraud Scenarios**

---

## Five Real-World Patterns Baked Into the Demo Data

<div class="small-text">

| # | Account | Fraud Pattern | Key Signals |
|---|---------|---------------|-------------|
| 1 | ACC003 | **Card Testing Attack** | 12 micro-charges then $899.99 at blacklisted merchant |
| 2 | ACC001 | **High-Value Night Transaction** | $4,500 at 2:47 AM, blacklisted, international |
| 3 | ACC002 | **Geographic Anomaly** | Lagos & Moscow vs Chicago baseline |
| 4 | ACC004 | **Velocity Attack** | 18 txns in 25.5 min, 5 high-risk merchants |
| 5 | ACC007 | **Suspicious Merchant** | $1,200 + $750 at QuickLoans.net at midnight |

</div>

All data is fully synthetic — `random.seed(42)` ensures reproducibility.

---

<!-- _class: lead -->

# **✨ Key Differentiators**

---

## What Makes 🛡️ _FraudGuard AI_ Different

### 1. Hybrid Intelligence
IsolationForest provides instant 0–100 score with anomaly flags; LLM agent reasons across 10 tools to explain *why* it looks fraudulent

### 2. Human-in-the-Loop Safety
The agent **recommends**, the analyst **approves** — no autonomous account freezes

### 3. Full Reasoning Transparency
Every tool call logged with inputs/outputs; complete LLM chain-of-thought captured

---

## What Makes 🛡️ _FraudGuard AI_ Different (continued)

### 4. Zero-Cost Inference
- LLM: `nvidia/nemotron-3-super-120b-a12b:free` on OpenRouter free tier
- Embeddings: `all-mpnet-base-v2` runs locally
- **Total inference cost: $0**

### 5. No Labelled Fraud Data Required
IsolationForest is unsupervised — learns normal patterns and flags deviations. Deploy on any institution's transaction history without expensive fraud labels.

---

<!-- _class: lead -->

# **🎯 Conclusion**

---

<!-- _class: lead -->

## **🛡️ _FraudGuard AI_ turns a 4-hour manual review into an AI investigation completed in minutes — with a human safety gate before any irreversible action.**

---

<!-- _class: lead -->

# The future of fraud prevention isn't more rules.

# It's **intelligent agents that reason, with humans who decide.**

---

<!-- _class: lead -->
<!-- _paginate: false -->

# **Thank You!**

**🛡️ _FraudGuard AI_**
Built at GrabHack 2.0 | May 2026

by Team Runtime Terror
