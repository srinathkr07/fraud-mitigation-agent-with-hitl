"""
backend/database.py
===================
SQLite persistence layer for fraud mitigation actions.

Stores every FLAG_TRANSACTION and FREEZE_ACCOUNT action committed through
the HITL panels, together with the analyst's decision and notes.
Also tracks current account freeze status so the agent can query it.

DB file: backend/fraud_actions.db (created automatically on first run).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "fraud_actions.db"

_CREATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS actions (
    action_id        TEXT PRIMARY KEY,
    action_type      TEXT NOT NULL,
    transaction_id   TEXT,
    account_id       TEXT NOT NULL,
    account_holder   TEXT,
    reason           TEXT,
    risk_score       INTEGER,
    agent_verdict    TEXT,
    analyst_decision TEXT,
    analyst_notes    TEXT,
    investigation_id TEXT,
    created_at       TEXT NOT NULL
);
"""

_CREATE_ACCOUNT_STATUS = """
CREATE TABLE IF NOT EXISTS account_status (
    account_id       TEXT PRIMARY KEY,
    holder_name      TEXT,
    is_frozen        INTEGER NOT NULL DEFAULT 0,
    freeze_reason    TEXT,
    freeze_action_id TEXT,
    frozen_at        TEXT,
    last_updated     TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = _connect()
    try:
        conn.execute(_CREATE_ACTIONS)
        conn.execute(_CREATE_ACCOUNT_STATUS)
        conn.commit()
    finally:
        conn.close()


def log_action(
    entry: dict,
    analyst_decision: str,
    analyst_notes: str,
    risk_score: int | None,
    agent_verdict: str,
    investigation_id: str,
) -> str:
    """
    Insert one FLAG_TRANSACTION or FREEZE_ACCOUNT entry into `actions`.
    For FREEZE_ACCOUNT, also upserts `account_status` (is_frozen=1).
    Returns the new action_id (UUID4).
    """
    action_type = entry.get("action")
    if not action_type:
        raise ValueError(f"log_action: entry missing required 'action' key: {entry!r}")

    action_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO actions (
                action_id, action_type, transaction_id, account_id, account_holder,
                reason, risk_score, agent_verdict, analyst_decision, analyst_notes,
                investigation_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_id,
                action_type,
                entry.get("transaction_id"),
                entry.get("account_id", ""),
                entry.get("account_holder"),
                entry.get("reason"),
                risk_score,
                agent_verdict,
                analyst_decision,
                analyst_notes or None,
                investigation_id,
                entry.get("timestamp", now),
            ),
        )

        if action_type == "FREEZE_ACCOUNT":
            conn.execute(
                """
                INSERT INTO account_status
                    (account_id, holder_name, is_frozen, freeze_reason,
                     freeze_action_id, frozen_at, last_updated)
                VALUES (?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    holder_name      = excluded.holder_name,
                    is_frozen        = 1,
                    freeze_reason    = excluded.freeze_reason,
                    freeze_action_id = excluded.freeze_action_id,
                    frozen_at        = excluded.frozen_at,
                    last_updated     = excluded.last_updated
                """,
                (
                    entry.get("account_id", ""),
                    entry.get("account_holder"),
                    entry.get("reason"),
                    action_id,
                    entry.get("timestamp", now),
                    now,
                ),
            )

        conn.commit()
    finally:
        conn.close()

    return action_id


def unfreeze_account(account_id: str) -> None:
    """Clear the freeze state for an account and log the UNFREEZE action."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT holder_name, is_frozen, freeze_reason, freeze_action_id
            FROM account_status WHERE account_id = ?
            """,
            (account_id,),
        ).fetchone()

        if row is None or not row["is_frozen"]:
            return  # account not frozen — nothing to do, no phantom audit entry

        holder_name = row["holder_name"]
        prior_reason = row["freeze_reason"]
        prior_action_id = row["freeze_action_id"]

        conn.execute(
            """
            INSERT INTO actions (
                action_id, action_type, transaction_id, account_id, account_holder,
                reason, risk_score, agent_verdict, analyst_decision, analyst_notes,
                investigation_id, created_at
            ) VALUES (?, 'UNFREEZE_ACCOUNT', NULL, ?, ?, ?, NULL, NULL, 'analyst', NULL, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                account_id,
                holder_name,
                f"Manually unfrozen. Prior freeze reason: {prior_reason or 'N/A'}",
                prior_action_id or "N/A",
                now,
            ),
        )

        conn.execute(
            """
            UPDATE account_status
            SET is_frozen=0, freeze_reason=NULL, freeze_action_id=NULL,
                frozen_at=NULL, last_updated=?
            WHERE account_id=?
            """,
            (now, account_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_account_freeze_status(account_id: str) -> dict | None:
    """Return the current account_status row for account_id, or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM account_status WHERE account_id = ?", (account_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_all_actions(limit: int = 200) -> list[dict]:
    """Return up to `limit` actions, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM actions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_frozen_accounts() -> list[dict]:
    """Return all accounts currently frozen."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM account_status WHERE is_frozen = 1 ORDER BY frozen_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_flagged_transactions() -> list[dict]:
    """Return all FLAG_TRANSACTION actions, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM actions WHERE action_type = 'FLAG_TRANSACTION' ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
