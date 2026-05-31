"""
tools.py
========
LangChain tools exposed to the fraud agent.

All tools are created via `create_fraud_tools(...)` which closes over
the shared data/model objects injected at startup — no global state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.tools import tool as _tool_decorator

if TYPE_CHECKING:
    from ml.anomaly_detector import FraudDetector
    from agent.memory_store import FraudVectorMemory


# ── Tool factory ──────────────────────────────────────────────────────────────

def create_fraud_tools(
    transactions_df: pd.DataFrame,
    accounts: dict,
    detector: "FraudDetector",
    memory: "FraudVectorMemory",
    pending_actions: list,
):
    """
    Return a list of LangChain tools wired to the provided data sources.

    Parameters
    ----------
    transactions_df : full mock transaction DataFrame
    accounts        : ACCOUNTS dict from mock_data
    detector        : trained FraudDetector instance
    memory          : built FraudVectorMemory instance
    pending_actions : mutable list that queues protective actions for human review
    """

    # ── Helpers (private to this factory scope) ───────────────────────────────
    def _get_txn(tid: str) -> dict | None:
        row = transactions_df[transactions_df["transaction_id"] == tid]
        return None if row.empty else row.iloc[0].to_dict()

    # ── Tool 1: Get transaction details ───────────────────────────────────────
    @_tool_decorator
    def get_transaction_details(transaction_id: str) -> str:
        """
        Retrieve complete details of a specific transaction by its ID.
        Always call this FIRST before any other analysis.
        Returns: amount, merchant, category, risk level, location, timestamp, account info.
        """
        txn = _get_txn(transaction_id)
        if txn is None:
            return f"ERROR: Transaction '{transaction_id}' not found."

        ts = pd.Timestamp(txn["timestamp"])
        acc = accounts.get(txn["account_id"], {})
        day_name = ts.strftime("%A")
        time_str = ts.strftime("%I:%M %p")
        is_night = ts.hour < 6 or ts.hour >= 22

        return (
            f"Transaction ID    : {txn['transaction_id']}\n"
            f"Account           : {txn['account_id']} ({acc.get('name', 'Unknown')})\n"
            f"Amount            : ${txn['amount']:,.2f}\n"
            f"Merchant          : {txn['merchant']}\n"
            f"Category          : {txn['category']}\n"
            f"Merchant Risk     : {txn['merchant_risk_level'].upper()}\n"
            f"Location          : {txn['location']}\n"
            f"International     : {'YES ⚠' if txn['is_international'] else 'No'}\n"
            f"Timestamp         : {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Day / Time        : {day_name}, {time_str}"
            f"{' ⚠ LATE NIGHT' if is_night else ''}\n"
            f"Account Avg Spend : ${acc.get('avg_amount', 0):,.2f}\n"
            f"Daily Limit       : ${acc.get('daily_limit', 0):,.2f}"
        )

    # ── Tool 2: ML risk score ─────────────────────────────────────────────────
    @_tool_decorator
    def analyze_ml_risk_score(transaction_id: str) -> str:
        """
        Run the IsolationForest ML model on a transaction to get an anomaly
        risk score (0–100) and a list of triggered anomaly flags.
        Higher score = higher fraud risk.
        Flags include: HIGH_AMOUNT, NIGHTTIME_TRANSACTION, INTERNATIONAL_TRANSACTION,
        HIGH_RISK_MERCHANT, HIGH_VELOCITY_1H, HIGH_VELOCITY_24H, UNUSUAL_AMOUNT_FOR_ACCOUNT.
        """
        result = detector.get_risk_assessment(transaction_id)
        if result["risk_score"] == -1:
            return f"ML model result: Unable to score transaction {transaction_id}."

        flags_str = (
            ", ".join(result["anomaly_flags"]) if result["anomaly_flags"] else "None"
        )
        return (
            f"ML Risk Score     : {result['risk_score']}/100\n"
            f"Risk Level        : {result['risk_level']}\n"
            f"Anomaly Flags     : {flags_str}\n"
            f"Explanation       : {result['explanation']}"
        )

    # ── Tool 3: Account history ───────────────────────────────────────────────
    @_tool_decorator
    def get_account_history(account_id: str) -> str:
        """
        Retrieve the last 30 days of transaction history for an account.
        Useful for spotting unusual spending patterns, new merchants, or deviations
        from the customer's normal behaviour.
        """
        from backend.database import get_account_freeze_status  # noqa: PLC0415

        acc = accounts.get(account_id)
        if acc is None:
            return f"ERROR: Account '{account_id}' not found."

        # Prepend live freeze status from the backend DB
        freeze_status = get_account_freeze_status(account_id)
        freeze_header = ""
        if freeze_status and freeze_status["is_frozen"]:
            freeze_header = (
                f"⚠ ACCOUNT CURRENTLY FROZEN\n"
                f"  Frozen since : {freeze_status['frozen_at']}\n"
                f"  Reason       : {freeze_status['freeze_reason']}\n\n"
            )

        max_ts = transactions_df[transactions_df["account_id"] == account_id]["timestamp"].max()
        cutoff = max_ts - pd.Timedelta(days=30)
        history = transactions_df[
            (transactions_df["account_id"] == account_id)
            & (transactions_df["timestamp"] >= cutoff)
        ].sort_values("timestamp", ascending=False)

        if history.empty:
            return freeze_header + f"No transactions found for {account_id} in the last 30 days."

        total_spend = history["amount"].sum()
        avg_spend = history["amount"].mean()
        fraud_count = int(history["is_fraud"].sum())
        intl_count = int(history["is_international"].sum())
        high_risk_count = int((history["merchant_risk_level"] == "high").sum())

        recent = history.head(8)
        txn_lines = "\n".join(
            f"  {row['timestamp'].strftime('%m/%d %H:%M')}  "
            f"${row['amount']:7.2f}  {row['merchant']:<25}  {row['location']}"
            for _, row in recent.iterrows()
        )

        return (
            freeze_header
            + f"Account           : {account_id} – {acc['name']}\n"
            f"Usual Locations   : {'; '.join(acc['usual_locations'])}\n"
            f"Account Avg Spend : ${acc['avg_amount']:,.2f}\n"
            f"--- Last 30 Days Summary ---\n"
            f"Total Transactions: {len(history)}\n"
            f"Total Spend       : ${total_spend:,.2f}\n"
            f"Average Amount    : ${avg_spend:,.2f}\n"
            f"International Txns: {intl_count}\n"
            f"High-Risk Merchant  : {high_risk_count}\n"
            f"Flagged Fraud Txns: {fraud_count}\n"
            f"--- Most Recent 8 Transactions ---\n"
            f"{txn_lines}"
        )

    # ── Tool 4: Merchant reputation ───────────────────────────────────────────
    @_tool_decorator
    def check_merchant_reputation(merchant_name: str) -> str:
        """
        Look up the risk profile and reputation of a merchant by name.
        Returns the merchant's risk level, category, and blacklist status.
        Use this to determine if a merchant is known to be associated with fraud.
        """
        from data.mock_data import MERCHANTS, MERCHANT_BLACKLIST  # noqa: PLC0415

        # Fuzzy match (case-insensitive)
        normalized = merchant_name.strip().lower()
        match = next(
            (name for name in MERCHANTS if name.lower() == normalized), None
        )
        if match is None:
            # Partial match
            match = next(
                (name for name in MERCHANTS if normalized in name.lower()), None
            )

        if match is None:
            return (
                f"Merchant '{merchant_name}' NOT FOUND in the merchant database.\n"
                f"This may indicate a ghost/unknown merchant — treat with caution.\n"
                f"Recommendation: FLAG for manual review."
            )

        info = MERCHANTS[match]
        is_blacklisted = match in MERCHANT_BLACKLIST
        risk_icon = {"low": "✓ LOW", "medium": "⚑ MEDIUM", "high": "✗ HIGH"}.get(
            info["risk_level"], f"? {info['risk_level'].upper()}"
        )

        lines = [
            f"Merchant          : {match}",
            f"Category          : {info['category']}",
            f"Risk Level        : {risk_icon}",
            f"Blacklisted       : {'YES — known fraudulent activity' if is_blacklisted else 'No'}",
        ]
        if is_blacklisted:
            lines.append("Action Recommended: BLOCK or ESCALATE immediately.")
        elif info["risk_level"] == "high":
            lines.append("Action Recommended: Flag for enhanced review.")

        return "\n".join(lines)

    # ── Tool 5: Transaction velocity ──────────────────────────────────────────
    @_tool_decorator
    def get_transaction_velocity(transaction_id: str) -> str:
        """
        Calculate how many transactions the account has made in the 1 hour, 24 hours,
        and 7 days up to and including the specified transaction (counts are inclusive
        of the transaction itself). High velocity is a key fraud indicator.
        Also reports the 7-day average daily transaction count as a baseline.
        Pass the transaction ID you are investigating.
        """
        txn = _get_txn(transaction_id)
        if txn is None:
            return f"ERROR: Transaction '{transaction_id}' not found."

        account_id = txn["account_id"]
        ref_ts = pd.Timestamp(txn["timestamp"])

        acc_txns = transactions_df[
            transactions_df["account_id"] == account_id
        ].sort_values("timestamp")

        if acc_txns.empty:
            return f"No transactions found for account {account_id}."

        last_1h = acc_txns[
            (acc_txns["timestamp"] >= ref_ts - pd.Timedelta(hours=1))
            & (acc_txns["timestamp"] <= ref_ts)
        ]
        last_24h = acc_txns[
            (acc_txns["timestamp"] >= ref_ts - pd.Timedelta(hours=24))
            & (acc_txns["timestamp"] <= ref_ts)
        ]
        last_7d = acc_txns[
            (acc_txns["timestamp"] >= ref_ts - pd.Timedelta(days=7))
            & (acc_txns["timestamp"] <= ref_ts)
        ]

        # Average daily transactions (baseline)
        acc = accounts.get(account_id, {})
        baseline_note = ""
        if len(last_7d) > 0:
            avg_daily = len(last_7d) / 7
            baseline_note = f"7-day Avg Daily   : {avg_daily:.1f} transactions/day"

        velocity_alert = ""
        if len(last_1h) > 3:
            velocity_alert = f"\n⚠ ALERT: {len(last_1h)} transactions in last hour — HIGH VELOCITY DETECTED"

        return (
            f"Account           : {account_id} ({acc.get('name', '')})\n"
            f"Transaction       : {transaction_id} at {ref_ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Last 1 Hour       : {len(last_1h)} transactions\n"
            f"Last 24 Hours     : {len(last_24h)} transactions\n"
            f"Last 7 Days       : {len(last_7d)} transactions\n"
            f"{baseline_note}"
            f"{velocity_alert}"
        )

    # ── Tool 6: Geographic anomaly ────────────────────────────────────────────
    @_tool_decorator
    def detect_geographic_anomaly(transaction_id: str) -> str:
        """
        Compare the specified transaction's location against the account's historical
        location pattern. Detects impossible travel (e.g., Chicago → Moscow in 2 hours)
        and transactions from countries the customer has never transacted from before.
        Pass the transaction ID you are investigating.
        """
        txn = _get_txn(transaction_id)
        if txn is None:
            return f"ERROR: Transaction '{transaction_id}' not found."

        account_id = txn["account_id"]
        acc = accounts.get(account_id)
        if acc is None:
            return f"ERROR: Account '{account_id}' not found."

        txn_ts = pd.Timestamp(txn["timestamp"])
        acc_txns = transactions_df[
            transactions_df["account_id"] == account_id
        ].sort_values("timestamp")

        if acc_txns.empty:
            return f"No transaction history for {account_id}."

        usual_locs = set(acc["usual_locations"])
        # Prior transactions = everything before this transaction's timestamp
        prior = acc_txns[acc_txns["timestamp"] < txn_ts]
        historical_locations = set(prior["location"].tolist()) if not prior.empty else set()
        all_known = usual_locs | historical_locations

        current_loc = txn["location"]
        is_intl = bool(txn["is_international"])

        anomaly_detected = current_loc not in all_known or is_intl

        # Check impossible travel: look at the transaction immediately before this one
        prev_alert = ""
        if not prior.empty:
            prev = prior.iloc[-1]
            time_diff = (txn_ts - pd.Timestamp(prev["timestamp"])).total_seconds() / 3600.0
            if prev["location"] != current_loc and time_diff < 2.0 and is_intl:
                prev_alert = (
                    f"\n⚠ IMPOSSIBLE TRAVEL DETECTED: "
                    f"Previous location '{prev['location']}' → Current '{current_loc}' "
                    f"in {time_diff:.1f} hours."
                )

        return (
            f"Account           : {account_id} ({acc.get('name', '')})\n"
            f"Transaction       : {transaction_id}\n"
            f"Known Locations   : {'; '.join(sorted(usual_locs))}\n"
            f"Transaction Location : {current_loc}\n"
            f"International     : {'YES' if is_intl else 'No'}\n"
            f"Anomaly Detected  : {'YES ⚠' if anomaly_detected else 'No'}"
            f"{prev_alert}"
        )

    # ── Tool 7: Search fraud patterns (vector memory) ─────────────────────────
    @_tool_decorator
    def search_fraud_patterns(description: str) -> str:
        """
        Semantic search over the library of known fraud patterns.
        Provide a natural-language description of the suspicious behaviour observed,
        and this tool returns the top matching known fraud types with mitigations.
        Example input: 'multiple rapid small transactions followed by large purchase'
        """
        results = memory.search(description, k=3)
        if not results:
            return "No matching fraud patterns found in memory."

        lines = [f"Top {len(results)} matching fraud patterns:\n"]
        for i, p in enumerate(results, 1):
            lines.append(
                f"[{i}] {p['name']} (Similarity: {p['similarity_score']:.2%}, Severity: {p['severity']})\n"
                f"    Description : {p['description']}\n"
                f"    Indicators  : {'; '.join(p['indicators'])}\n"
                f"    Mitigation  : {p['mitigation']}\n"
            )
        return "\n".join(lines)

    # ── Tool 8: Flag transaction ───────────────────────────────────────────────
    @_tool_decorator
    def flag_transaction(input_str: str) -> str:
        """
        Flag a transaction as suspicious and add it to the review queue.
        Input format: '<transaction_id> | <reason>'
        Example: 'TXN12345678 | High-value nighttime purchase at unknown merchant'
        """
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "ERROR: Use format '<transaction_id> | <reason>'"

        tid, reason = parts[0].strip(), parts[1].strip()
        txn = _get_txn(tid)
        if txn is None:
            return f"ERROR: Transaction '{tid}' not found."

        pending_actions.append(
            {
                "action": "FLAG_TRANSACTION",
                "transaction_id": tid,
                "account_id": txn["account_id"],
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "status": "PENDING",
            }
        )
        return (
            f"⏳ Transaction {tid} queued for review — pending analyst approval.\n"
            f"Account    : {txn['account_id']}\n"
            f"Amount     : ${txn['amount']:,.2f}\n"
            f"Reason     : {reason}\n"
            f"Status     : Awaiting human review before flagging."
        )

    # ── Tool 9: Freeze account ────────────────────────────────────────────────
    @_tool_decorator
    def freeze_account(input_str: str) -> str:
        """
        Freeze an account to prevent further transactions.
        Input format: '<account_id> | <reason>'
        Example: 'ACC003 | Velocity attack detected — 12 micro-transactions in 8 minutes'
        Use only when fraud confidence is HIGH or CRITICAL.
        """
        from backend.database import get_account_freeze_status  # noqa: PLC0415

        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "ERROR: Use format '<account_id> | <reason>'"

        acc_id, reason = parts[0].strip(), parts[1].strip()
        acc = accounts.get(acc_id)
        if acc is None:
            return f"ERROR: Account '{acc_id}' not found."

        existing = get_account_freeze_status(acc_id)
        if existing and existing["is_frozen"]:
            return (
                f"⚠ Account {acc_id} ({acc['name']}) is already frozen.\n"
                f"Frozen since : {existing['frozen_at']}\n"
                f"Reason       : {existing['freeze_reason']}\n"
                f"No duplicate action queued."
            )

        pending_actions.append(
            {
                "action": "FREEZE_ACCOUNT",
                "account_id": acc_id,
                "account_holder": acc["name"],
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "status": "PENDING",
            }
        )
        return (
            f"⏳ Account {acc_id} ({acc['name']}) freeze queued — pending analyst approval.\n"
            f"Reason     : {reason}\n"
            f"Status     : Awaiting human review. No action executed until approved."
        )

    # ── Tool 10: Recovery recommendations ────────────────────────────────────
    @_tool_decorator
    def get_recovery_recommendations(fraud_type: str) -> str:
        """
        Get practical, step-by-step recovery recommendations for a customer
        based on the type of fraud detected.
        Valid fraud_type values: card_testing, card_testing_followup, geo_anomaly,
        velocity_attack, high_value_night, suspicious_merchant, account_takeover, unknown.
        """
        recommendations = {
            "card_testing": [
                "1. Your card has been temporarily blocked to prevent further misuse.",
                "2. All micro-transactions (small test charges) have been flagged for reversal.",
                "3. Request a new card with a new card number — allow 3–5 business days.",
                "4. Review your statements for any transactions you don't recognise.",
                "5. Avoid saving card details on unverified websites.",
                "6. Consider enabling real-time transaction notifications.",
            ],
            "card_testing_followup": [
                "1. Large fraudulent purchase has been flagged and a dispute has been opened.",
                "2. Your card has been blocked — a replacement is being issued.",
                "3. Contact the merchant directly with your case reference number.",
                "4. File a formal fraud dispute with your bank within 60 days.",
                "5. Monitor your credit report for any new unauthorised credit enquiries.",
            ],
            "geo_anomaly": [
                "1. Transactions from unexpected international locations have been blocked.",
                "2. Your account is temporarily restricted to domestic transactions.",
                "3. Contact us immediately if you are currently travelling abroad.",
                "4. Change your online banking password and enable 2-factor authentication.",
                "5. Review all recent international transactions for unauthorised activity.",
                "6. Consider placing a travel notification before future international trips.",
            ],
            "velocity_attack": [
                "1. Your account has been frozen due to an unusual surge in transaction activity.",
                "2. All transactions during the attack window are under review.",
                "3. Contact our fraud team to verify which transactions are legitimate.",
                "4. A new card will be issued after your identity is verified.",
                "5. Review and revoke all third-party app access to your account.",
                "6. Update your PIN and online banking credentials immediately.",
            ],
            "high_value_night": [
                "1. This high-value transaction has been flagged and is under review.",
                "2. If you did not make this purchase, it will be reversed within 3–5 business days.",
                "3. Enable overnight spending limits through your banking app.",
                "4. Set up real-time SMS alerts for any transaction over a threshold you choose.",
                "5. Consider enabling biometric authentication for large transactions.",
            ],
            "suspicious_merchant": [
                "1. The transaction with the flagged merchant has been disputed.",
                "2. Do NOT share any additional personal or financial information with this merchant.",
                "3. If you signed up for a service, cancel it immediately and revoke card access.",
                "4. Monitor your account for recurring charges from this merchant.",
                "5. Report the merchant to the relevant consumer protection authority.",
            ],
            "account_takeover": [
                "1. Your account has been locked pending identity verification.",
                "2. Call our 24/7 fraud hotline immediately: 1-800-555-FRAUD.",
                "3. Reset all passwords — banking, email, and any linked services.",
                "4. Enable 2-factor authentication on all financial accounts.",
                "5. Review your credit report for any new accounts you did not open.",
                "6. Consider placing a fraud alert with all three credit bureaus.",
            ],
            "unknown": [
                "1. The flagged transaction is under investigation by our fraud team.",
                "2. You will be contacted within 24 hours with an update.",
                "3. In the meantime, monitor your account activity closely.",
                "4. Report any other suspicious charges immediately via your banking app.",
                "5. Keep your contact information up to date so we can reach you promptly.",
            ],
        }

        normalized = fraud_type.strip().lower().replace(" ", "_").replace("-", "_")
        steps = recommendations.get(normalized, recommendations["unknown"])

        header = f"Recovery Recommendations for '{fraud_type}':\n"
        return header + "\n".join(steps)

    return [
        get_transaction_details,
        analyze_ml_risk_score,
        get_account_history,
        check_merchant_reputation,
        get_transaction_velocity,
        detect_geographic_anomaly,
        search_fraud_patterns,
        flag_transaction,
        freeze_account,
        get_recovery_recommendations,
    ]
