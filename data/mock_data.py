"""
mock_data.py
============
Generates synthetic transaction data with embedded fraud patterns.
All data is deterministic (seeded) — no external API or database required.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

BASE_DATE = datetime(2024, 3, 1)
END_DATE = datetime(2024, 6, 1)

# ── Account profiles ──────────────────────────────────────────────────────────
ACCOUNTS: dict[str, dict] = {
    "ACC001": {
        "name": "Alice Johnson",
        "email": "alice.johnson@gmail.com",
        "phone": "+1-555-0101",
        "status": "active",
        "usual_locations": ["New York, NY", "Newark, NJ"],
        "avg_amount": 75.0,
        "daily_limit": 2000.0,
        "credit_limit": 5000.0,
        "member_since": "2019-06-15",
    },
    "ACC002": {
        "name": "Bob Martinez",
        "email": "bob.martinez@gmail.com",
        "phone": "+1-555-0202",
        "status": "active",
        "usual_locations": ["Chicago, IL", "Evanston, IL"],
        "avg_amount": 120.0,
        "daily_limit": 3000.0,
        "credit_limit": 8000.0,
        "member_since": "2018-01-20",
    },
    "ACC003": {
        "name": "Carol Williams",
        "email": "carol.williams@gmail.com",
        "phone": "+1-555-0303",
        "status": "active",
        "usual_locations": ["Los Angeles, CA", "Santa Monica, CA"],
        "avg_amount": 95.0,
        "daily_limit": 2500.0,
        "credit_limit": 6000.0,
        "member_since": "2020-09-10",
    },
    "ACC004": {
        "name": "David Chen",
        "email": "david.chen@gmail.com",
        "phone": "+1-555-0404",
        "status": "active",
        "usual_locations": ["Seattle, WA", "Bellevue, WA"],
        "avg_amount": 200.0,
        "daily_limit": 5000.0,
        "credit_limit": 12000.0,
        "member_since": "2017-03-05",
    },
    "ACC005": {
        "name": "Emma Davis",
        "email": "emma.davis@gmail.com",
        "phone": "+1-555-0505",
        "status": "active",
        "usual_locations": ["Miami, FL", "Fort Lauderdale, FL"],
        "avg_amount": 55.0,
        "daily_limit": 1500.0,
        "credit_limit": 3500.0,
        "member_since": "2021-11-30",
    },
    "ACC006": {
        "name": "Frank Wilson",
        "email": "frank.wilson@gmail.com",
        "phone": "+1-555-0606",
        "status": "active",
        "usual_locations": ["Boston, MA", "Cambridge, MA"],
        "avg_amount": 150.0,
        "daily_limit": 4000.0,
        "credit_limit": 10000.0,
        "member_since": "2016-07-22",
    },
    "ACC007": {
        "name": "Grace Lee",
        "email": "grace.lee@gmail.com",
        "phone": "+1-555-0707",
        "status": "active",
        "usual_locations": ["San Francisco, CA", "Oakland, CA"],
        "avg_amount": 180.0,
        "daily_limit": 4500.0,
        "credit_limit": 11000.0,
        "member_since": "2019-02-14",
    },
    "ACC008": {
        "name": "Henry Brown",
        "email": "henry.brown@gmail.com",
        "phone": "+1-555-0808",
        "status": "active",
        "usual_locations": ["Austin, TX", "Round Rock, TX"],
        "avg_amount": 110.0,
        "daily_limit": 3000.0,
        "credit_limit": 7000.0,
        "member_since": "2020-04-18",
    },
}

# ── Merchant catalog ──────────────────────────────────────────────────────────
MERCHANTS: dict[str, dict] = {
    "Walmart": {"category": "grocery", "risk_level": "low"},
    "Target": {"category": "retail", "risk_level": "low"},
    "Starbucks": {"category": "food", "risk_level": "low"},
    "McDonald's": {"category": "food", "risk_level": "low"},
    "Amazon": {"category": "online", "risk_level": "low"},
    "Netflix": {"category": "subscription", "risk_level": "low"},
    "Whole Foods": {"category": "grocery", "risk_level": "low"},
    "Uber": {"category": "transport", "risk_level": "low"},
    "CVS Pharmacy": {"category": "pharmacy", "risk_level": "low"},
    "Gym & Fitness Club": {"category": "fitness", "risk_level": "low"},
    "Shell Gas": {"category": "gas", "risk_level": "medium"},
    "Best Buy": {"category": "electronics", "risk_level": "low"},
    "ATM Withdrawal": {"category": "cash", "risk_level": "medium"},
    "Delta Airlines": {"category": "travel", "risk_level": "medium"},
    "Marriott Hotels": {"category": "travel", "risk_level": "medium"},
    "CryptoExchange Pro": {"category": "crypto", "risk_level": "high"},
    "QuickLoans.net": {"category": "financial", "risk_level": "high"},
    "ZzaapDeals.com": {"category": "unknown", "risk_level": "high"},
    "Offshore Store Ltd": {"category": "unknown", "risk_level": "high"},
    "ElectroGear Pro": {"category": "electronics", "risk_level": "high"},
}

_LOW_RISK = [m for m, d in MERCHANTS.items() if d["risk_level"] == "low"]
_MED_RISK = [m for m, d in MERCHANTS.items() if d["risk_level"] == "medium"]
_HIGH_RISK = [m for m, d in MERCHANTS.items() if d["risk_level"] == "high"]

MERCHANT_BLACKLIST = {"CryptoExchange Pro", "ZzaapDeals.com", "Offshore Store Ltd"}

# ── Known fraud patterns (ingested into FAISS vector store) ──────────────────
KNOWN_FRAUD_PATTERNS: list[dict] = [
    {
        "id": "FP001",
        "name": "Card Testing Attack",
        "description": (
            "Fraudster uses stolen card details to make multiple small test transactions "
            "(often $0.50–$5.00) in rapid succession across different merchants to verify "
            "card validity before committing larger fraud purchases."
        ),
        "indicators": [
            "multiple transactions under $5 within minutes",
            "high transaction velocity (>5 in 10 minutes)",
            "varied merchants in quick succession",
            "followed by a large purchase",
        ],
        "severity": "HIGH",
        "mitigation": (
            "Flag card for analyst review. Flag all transactions in the testing window for reversal."
            "Contact customer to confirm fraud. Issue replacement card."
        ),
    },
    {
        "id": "FP002",
        "name": "Card Skimming / POS Compromise",
        "description": (
            "Physical card data stolen via compromised POS terminal or ATM skimmer. "
            "Often followed by cloned card transactions at different locations shortly after."
        ),
        "indicators": [
            "transaction immediately after ATM or gas station use",
            "simultaneous transactions in geographically distant locations",
            "card-present transaction in unusual location",
        ],
        "severity": "HIGH",
        "mitigation": (
            "Block card. Identify compromised terminal. Review all recent transactions. "
            "Issue replacement card with new number."
        ),
    },
    {
        "id": "FP003",
        "name": "Account Takeover Fraud",
        "description": (
            "Criminal gains unauthorized access via phishing or credential stuffing, "
            "changes account settings then drains funds or makes large purchases."
        ),
        "indicators": [
            "recent password or email change before large transaction",
            "login from unusual device or IP",
            "unusual large purchase at unknown or high-risk merchant",
            "change of contact information",
        ],
        "severity": "CRITICAL",
        "mitigation": (
            "Freeze account pending analyst approval. Require identity verification. "
            "Review all device sessions. Reverse unauthorized transactions."
        ),
    },
    {
        "id": "FP004",
        "name": "Geographic Anomaly / International Fraud",
        "description": (
            "Transaction occurs in a country or location significantly different from the "
            "customer's known pattern, suggesting card data theft or account compromise."
        ),
        "indicators": [
            "transaction in foreign country with no prior international history",
            "multiple countries in short timeframe (impossible travel)",
            "unusual location compared to established account history",
        ],
        "severity": "HIGH",
        "mitigation": (
            "Decline or flag transaction. Contact customer immediately. "
            "If unrecognized, block card and initiate dispute process."
        ),
    },
    {
        "id": "FP005",
        "name": "High-Value Nighttime Transaction",
        "description": (
            "Unusually large purchases made between 10 PM and 6 AM for accounts "
            "with no prior late-night activity pattern."
        ),
        "indicators": [
            "transaction between 10 PM and 6 AM (outside normal hours)",
            "amount significantly above account average",
            "first-time merchant or unusual category",
        ],
        "severity": "MEDIUM",
        "mitigation": (
            "Flag for review. Send real-time alert to customer. "
            "Require step-up authentication before approving."
        ),
    },
    {
        "id": "FP006",
        "name": "Velocity Attack",
        "description": (
            "Large number of transactions made in a very short period, indicating "
            "automated fraud tools or compromised credentials being exploited at scale."
        ),
        "indicators": [
            "more than 10 transactions within 30 minutes",
            "consistent or slightly varying transaction amounts",
            "targeting online or high-risk merchants",
        ],
        "severity": "CRITICAL",
        "mitigation": (
            "Flag all transactions for analyst review. Queue account freeze pending analyst approval. "
            "Contact customer. Escalate to fraud investigation team."
        ),
    },
    {
        "id": "FP007",
        "name": "Suspicious / Ghost Merchant",
        "description": (
            "Transaction at a merchant flagged on watchlists, or whose category "
            "is inconsistent with the customer's spending profile."
        ),
        "indicators": [
            "merchant on fraud watchlist or blacklist",
            "high-risk category (crypto, predatory financial services)",
            "merchant in unexpected or implausible location",
        ],
        "severity": "HIGH",
        "mitigation": (
            "Flag transaction. Risk-score the merchant. Contact customer if confirmed. "
            "Initiate chargeback if unauthorized."
        ),
    },
    {
        "id": "FP008",
        "name": "Friendly Fraud / Chargeback Fraud",
        "description": (
            "Legitimate cardholder makes purchase then falsely disputes it as unauthorized "
            "to obtain a refund while retaining goods or services."
        ),
        "indicators": [
            "customer has prior dispute history",
            "delivery confirmed but dispute filed",
            "digital goods or non-refundable service",
        ],
        "severity": "MEDIUM",
        "mitigation": (
            "Gather transaction evidence. Compare with dispute history. "
            "Coordinate with merchant for delivery proof."
        ),
    },
]

# ── Internal helpers ──────────────────────────────────────────────────────────
_HOUR_WEIGHTS = [
    1, 1, 1, 1, 1, 2,   # 00-05 (night)
    4, 8, 10, 10, 9, 9, # 06-11 (morning)
    8, 9, 9, 10, 10, 9, # 12-17 (afternoon)
    8, 7, 6, 5, 3, 2,   # 18-23 (evening)
]


def _rand_ts(start: datetime = BASE_DATE, end: datetime = END_DATE) -> datetime:
    delta = int((end - start).total_seconds())
    base = start + timedelta(seconds=random.randint(0, delta))
    hour = random.choices(range(24), weights=_HOUR_WEIGHTS)[0]
    return base.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def _make_id() -> str:
    return "TXN" + uuid.uuid4().hex[:8].upper()


# ── Normal transaction generator ──────────────────────────────────────────────
def _normal_transactions(account_id: str, account: dict, n: int) -> list[dict]:
    records = []
    for _ in range(n):
        amount = round(float(np.random.lognormal(np.log(account["avg_amount"]), 0.55)), 2)
        amount = max(1.50, min(amount, account["daily_limit"] * 0.65))

        r = random.random()
        if r < 0.75:
            merchant = random.choice(_LOW_RISK)
        elif r < 0.95:
            merchant = random.choice(_MED_RISK)
        else:
            # Remaining 5% also maps to low-risk to keep normal transactions
            # cleanly separable from fraud scenarios (no high-risk merchants).
            merchant = random.choice(_LOW_RISK[:5])

        records.append(
            {
                "transaction_id": _make_id(),
                "account_id": account_id,
                "timestamp": _rand_ts(),
                "amount": amount,
                "merchant": merchant,
                "merchant_risk_level": MERCHANTS[merchant]["risk_level"],
                "category": MERCHANTS[merchant]["category"],
                "location": random.choice(account["usual_locations"]),
                "is_international": False,
                "is_fraud": False,
                "fraud_type": None,
            }
        )
    return records


# ── Fraud scenario generators ─────────────────────────────────────────────────
def _fraud_card_testing() -> list[dict]:
    """ACC003 — 12 micro-transactions then a large purchase."""
    records = []
    base = datetime(2024, 4, 15, 11, 22, 0)
    micro_merchants = ["PayPal Micro", "Steam Store", "Google Play", "Apple iTunes"]
    for i in range(12):
        tid = _make_id()
        records.append(
            {
                "transaction_id": tid,
                "account_id": "ACC003",
                "timestamp": base + timedelta(seconds=i * 38),
                "amount": round(random.uniform(0.50, 4.99), 2),
                "merchant": random.choice(micro_merchants),
                "merchant_risk_level": "medium",
                "category": "online",
                "location": "Online",
                "is_international": False,
                "is_fraud": True,
                "fraud_type": "card_testing",
            }
        )
    records.append(
        {
            "transaction_id": _make_id(),
            "account_id": "ACC003",
            "timestamp": base + timedelta(minutes=11),
            "amount": 899.99,
            "merchant": "Offshore Store Ltd",
            "merchant_risk_level": "high",
            "category": "unknown",
            "location": "Online (Unknown)",
            "is_international": True,
            "is_fraud": True,
            "fraud_type": "card_testing_followup",
        }
    )
    return records


def _fraud_night_transaction() -> list[dict]:
    """ACC001 — $4 500 purchase at 2:47 AM."""
    return [
        {
            "transaction_id": _make_id(),
            "account_id": "ACC001",
            "timestamp": datetime(2024, 5, 3, 2, 47, 0),
            "amount": 4500.00,
            "merchant": "ZzaapDeals.com",
            "merchant_risk_level": "high",
            "category": "unknown",
            "location": "Online",
            "is_international": True,
            "is_fraud": True,
            "fraud_type": "high_value_night",
        }
    ]


def _fraud_geo_anomaly() -> list[dict]:
    """ACC002 — transactions from Nigeria and Russia."""
    return [
        {
            "transaction_id": _make_id(),
            "account_id": "ACC002",
            "timestamp": datetime(2024, 4, 30, 15, 30, 0),
            "amount": 350.00,
            "merchant": "ElectroGear Pro",
            "merchant_risk_level": "high",
            "category": "electronics",
            "location": "Lagos, Nigeria",
            "is_international": True,
            "is_fraud": True,
            "fraud_type": "geo_anomaly",
        },
        {
            "transaction_id": _make_id(),
            "account_id": "ACC002",
            "timestamp": datetime(2024, 4, 30, 17, 15, 0),
            "amount": 820.00,
            "merchant": "Offshore Store Ltd",
            "merchant_risk_level": "high",
            "category": "unknown",
            "location": "Moscow, Russia",
            "is_international": True,
            "is_fraud": True,
            "fraud_type": "geo_anomaly",
        },
    ]


def _fraud_velocity_attack() -> list[dict]:
    """ACC004 — 18 transactions in 25.5 minutes across high-risk merchants."""
    records = []
    base = datetime(2024, 4, 22, 14, 30, 0)
    attack_merchants = [
        "CryptoExchange Pro",
        "QuickLoans.net",
        "ZzaapDeals.com",
        "Offshore Store Ltd",
        "ElectroGear Pro",
    ]
    for i in range(18):
        records.append(
            {
                "transaction_id": _make_id(),
                "account_id": "ACC004",
                "timestamp": base + timedelta(seconds=i * 90),
                "amount": round(random.uniform(50.00, 280.00), 2),
                "merchant": random.choice(attack_merchants),
                "merchant_risk_level": "high",
                "category": random.choice(["crypto", "financial", "unknown"]),
                "location": random.choice(["Online (VPN)", "Online (Unknown)", "Online"]),
                "is_international": random.choice([True, True, False]),
                "is_fraud": True,
                "fraud_type": "velocity_attack",
            }
        )
    return records


def _fraud_suspicious_merchant() -> list[dict]:
    """ACC007 — late-night transactions at high-risk financial/crypto merchants (CryptoExchange Pro is blacklisted; QuickLoans.net is high-risk only)."""
    return [
        {
            "transaction_id": _make_id(),
            "account_id": "ACC007",
            "timestamp": datetime(2024, 5, 10, 23, 55, 0),
            "amount": 1200.00,
            "merchant": "QuickLoans.net",
            "merchant_risk_level": "high",
            "category": "financial",
            "location": "Online",
            "is_international": False,
            "is_fraud": True,
            "fraud_type": "suspicious_merchant",
        },
        {
            "transaction_id": _make_id(),
            "account_id": "ACC007",
            "timestamp": datetime(2024, 5, 11, 0, 12, 0),
            "amount": 750.00,
            "merchant": "CryptoExchange Pro",
            "merchant_risk_level": "high",
            "category": "crypto",
            "location": "Online",
            "is_international": True,
            "is_fraud": True,
            "fraud_type": "suspicious_merchant",
        },
    ]


# ── Main generator (cached) ───────────────────────────────────────────────────
def generate_transactions() -> pd.DataFrame:
    """Return the full synthetic transaction dataset."""
    records: list[dict] = []
    for acc_id, acc in ACCOUNTS.items():
        records.extend(_normal_transactions(acc_id, acc, 350))

    records.extend(_fraud_card_testing())
    records.extend(_fraud_night_transaction())
    records.extend(_fraud_geo_anomaly())
    records.extend(_fraud_velocity_attack())
    records.extend(_fraud_suspicious_merchant())

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
