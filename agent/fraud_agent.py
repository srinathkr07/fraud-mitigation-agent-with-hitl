"""
fraud_agent.py
==============
Core LangChain agent for fraud investigation.

Uses `langchain.agents.create_agent` with native tool-calling.
The agent receives a high-level goal, calls tools in a loop using native tool-calling,
and produces a structured fraud report.

LLM: nvidia/nemotron-3-super-120b-a12b:free via OpenRouter (langchain-openrouter).
      120B-param hybrid Mamba-Transformer MoE model (12B active), native tool use, 262K context.
Framework: LangChain create_agent (tool-calling loop).
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are FraudGuard AI, an expert fraud detection agent at a financial institution.
Your mission: thoroughly investigate suspicious transactions and protect customers from financial harm.

INVESTIGATION PROTOCOL — follow this order every time:
1. get_transaction_details      — retrieve full transaction info (ALWAYS first)
2. analyze_ml_risk_score        — get the ML anomaly score and flags
3. get_account_history          — review the customer's recent spending pattern
4. check_merchant_reputation    — verify whether the merchant is flagged
5. get_transaction_velocity     — check for unusual frequency/bursts (pass the transaction ID)
6. detect_geographic_anomaly    — look for impossible travel or unusual locations (pass the transaction ID)
7. search_fraud_patterns        — match observed behaviour to known fraud types
8. flag_transaction             — queue the transaction for analyst review if risk is HIGH or CRITICAL (action is PENDING until human approval)
9. freeze_account               — queue an account freeze for analyst review if risk is HIGH or CRITICAL (action is PENDING until human approval)
10. get_recovery_recommendations — always retrieve steps for the customer

VERDICT SCALE:
  LEGITIMATE  — risk score < 35, no major anomaly flags
  SUSPICIOUS  — risk score 35–69, warrants monitoring
  FRAUDULENT  — risk score ≥ 70 OR critical anomaly flags present

RISK LEVEL REFERENCE (from ML model):
  LOW      — risk score 0–34
  MEDIUM   — risk score 35–69
  HIGH     — risk score 70–79
  CRITICAL — risk score 80–100

When the ML model returns HIGH or CRITICAL risk levels, treat both as requiring immediate attention and consider flagging/freezing actions.

After completing all tool calls, respond with ONLY the following structured report. Do not add any text before or after it:

---
**VERDICT**: [LEGITIMATE | SUSPICIOUS | FRAUDULENT]
**Risk Score**: [X/100]
**Fraud Type**: [detected fraud pattern or "N/A"]
**Key Risk Factors**:
  - [factor 1]
  - [factor 2]
**Actions Taken**:
  - [action 1 or "None required"]
**Recovery Steps for Customer**:
  - [step 1]
  - [step 2]
---"""


class FraudMitigationAgent:
    """
    Wraps a LangChain create_agent for fraud investigation.

    Usage::

        agent = FraudMitigationAgent(api_key="sk-...", tools=tools)
        result = agent.investigate("TXN12345678")
        print(result["final_answer"])
        for step in result["reasoning_steps"]:
            print(step)
    """

    def __init__(self, api_key: str, tools: list) -> None:
        llm = ChatOpenRouter(
            model_name="nvidia/nemotron-3-super-120b-a12b:free",
            openrouter_api_key=api_key,
            max_tokens=8192,
            max_retries=2,
            reasoning={"effort": "medium", "summary": "auto"},
            app_title="FraudGuard AI",
        )

        self._graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
        )

    # ── Public interface ──────────────────────────────────────────────────────
    def investigate(self, transaction_id: str) -> dict[str, Any]:
        """
        Run a full fraud investigation on the given transaction.

        Returns
        -------
        dict with keys:
            transaction_id    : str
            final_answer      : str  (structured report)
            reasoning_steps   : list[dict]   each has 'reasoning', 'thought', 'action', 'action_input', 'observation'
            verdict           : str  ("LEGITIMATE" | "SUSPICIOUS" | "FRAUDULENT" | "UNKNOWN")
            risk_score        : int | None
            elapsed_seconds   : float
            error             : str | None
            error_type        : str | None  ("auth_error" | "rate_limit" | "quota_exhausted" | "connection_error" | "unknown_error")
        """
        question = (
            f"Investigate transaction {transaction_id} for potential fraud. "
            f"Follow the full investigation protocol and produce a structured final report."
        )

        start = time.time()
        error = None
        error_type = None
        final_answer = "No output produced."
        all_messages: list = []

        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=question)]}
            )
            all_messages = result.get("messages", [])
            # The last AIMessage without tool_calls is the final answer
            final_answer = self._extract_final_answer(all_messages)
        except Exception as exc:  # noqa: BLE001
            error_type, error = self._classify_error(exc)

        elapsed = round(time.time() - start, 2)
        reasoning_steps = self._parse_messages(all_messages)
        verdict = self._extract_verdict(final_answer)
        risk_score = self._extract_risk_score(final_answer)

        return {
            "transaction_id": transaction_id,
            "final_answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "verdict": verdict,
            "risk_score": risk_score,
            "elapsed_seconds": elapsed,
            "error": error,
            "error_type": error_type,
        }

    # ── Private helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _classify_error(exc: Exception) -> tuple[str, str]:
        """Return (error_type, user_message) for known OpenRouter HTTP errors."""
        msg = str(exc)
        if "401" in msg or "403" in msg:
            return "auth_error", "Invalid or missing API key — check your OPENROUTER_API_KEY."
        if "429" in msg:
            return "rate_limit", "Rate limit reached — too many requests. Wait a moment and try again."
        if "402" in msg or "529" in msg:
            return "quota_exhausted", "Daily free-tier quota on OpenRouter is exhausted. Try again tomorrow."
        if any(kw in msg.lower() for kw in ("connection", "timeout", "network", "unreachable")):
            return "connection_error", "Cannot reach the OpenRouter API. Check your internet connection."
        return "unknown_error", msg

    @staticmethod
    def _extract_final_answer(messages: list) -> str:
        """Return the last AI text response that contains no tool calls."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    continue
                # content_blocks — unofficial attribute probed at runtime; present on some
                # ChatOpenRouter responses to expose reasoning and text blocks separately
                content_blocks = getattr(msg, "content_blocks", None) or []
                text_from_blocks = " ".join(
                    b.get("text", "") for b in content_blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
                if text_from_blocks:
                    return text_from_blocks
                # Fallback to msg.content
                content = msg.content
                if isinstance(content, list):
                    parts = [
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    ]
                    content = "".join(parts)
                if content and content.strip():
                    return content.strip()
        return "Agent did not produce a final report."

    @staticmethod
    def _parse_messages(messages: list) -> list[dict]:
        """
        Convert the message list into the reasoning-step dicts
        expected by the UI and log-saving code.

        Each tool call becomes one step dict with:
            reasoning      — model chain-of-thought text (from content_blocks, if present)
            thought        — AI text content before the call (if any)
            action         — tool name
            action_input   — tool input string
            observation    — tool output string
        """
        steps: list[dict] = []
        # Build a lookup: tool_call_id → ToolMessage content
        tool_results: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_results[msg.tool_call_id] = str(msg.content)

        for msg in messages:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue

            # 1. content_blocks — native ChatOpenRouter attribute with reasoning blocks
            thought_text = ""
            reasoning_text = ""
            content_blocks = getattr(msg, "content_blocks", None) or []
            if content_blocks:
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        thought_text += block.get("text", "")
                    elif btype == "reasoning":
                        reasoning_text += block.get("reasoning", block.get("text", ""))
                thought_text = thought_text.strip()
                reasoning_text = reasoning_text.strip()

            # 2. Fallback: read from msg.content list / string
            if not thought_text and not reasoning_text:
                if isinstance(msg.content, str):
                    thought_text = msg.content.strip()
                elif isinstance(msg.content, list):
                    text_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    for block in msg.content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "text":
                                text_parts.append(block.get("text", ""))
                            elif btype in ("thinking", "reasoning"):
                                reasoning_parts.append(
                                    block.get("thinking", block.get("reasoning", block.get("text", "")))
                                )
                    thought_text = "".join(text_parts).strip()
                    reasoning_text = "".join(reasoning_parts).strip()

            # 3. Last resort: additional_kwargs (some OpenRouter models)
            if not reasoning_text:
                reasoning_text = (
                    msg.additional_kwargs.get("reasoning_content", "") or ""
                ).strip()

            for tc in msg.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                call_id   = tc.get("id", "")

                # Flatten single-arg tools to a plain string for display
                if isinstance(tool_args, dict) and len(tool_args) == 1:
                    action_input = str(next(iter(tool_args.values())))
                else:
                    action_input = str(tool_args)

                observation = tool_results.get(call_id, "")

                steps.append({
                    "thought": thought_text,
                    "reasoning": reasoning_text,
                    "action": tool_name,
                    "action_input": action_input,
                    "observation": observation,
                })
                # Only attach thought/reasoning to the first call in this message
                thought_text = ""
                reasoning_text = ""

        return steps

    @staticmethod
    def _extract_verdict(text: str) -> str:
        m = re.search(
            r'\*{0,2}VERDICT\*{0,2}\s*:\s*(FRAUDULENT|SUSPICIOUS|LEGITIMATE)',
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()
        for word in ("FRAUDULENT", "SUSPICIOUS", "LEGITIMATE"):
            if word == "FRAUDULENT" and re.search(r'\bnot\s+fraudulent\b', text, re.IGNORECASE):
                continue
            if word in text.upper():
                return word
        return "UNKNOWN"

    @staticmethod
    def _extract_risk_score(text: str) -> int | None:
        match = re.search(r"Risk Score[^\d]*(\d{1,3})\s*/\s*100", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
