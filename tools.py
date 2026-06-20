"""
Tools available to the LLM bot.
"""

import csv
import json
import logging
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from manifold import ManifoldClient

logger = logging.getLogger(__name__)

PLAN_FILE = Path("PLAN.md")
TRADES_FILE = Path("trades.csv")


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo (free, no API key required).

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        JSON string with search results.
    """
    logger.info(f"Web search: '{query}'")
    try:
        with DDGS(timeout=30) as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)
                if len(results) >= max_results:
                    break

        if not results:
            return json.dumps({"results": [], "message": "No results found."})

        formatted = []
        for r in results:
            formatted.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
            )

        result_str = json.dumps({"results": formatted, "count": len(formatted)})
        logger.info(
            f"Web search returned {len(formatted)} results ({(len(result_str))} chars)"
        )
        return result_str

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return json.dumps({"error": str(e), "results": []})


def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract readable text content from a URL.

    Args:
        url: The URL to fetch content from.
        max_chars: Maximum characters to return (default 8000).

    Returns:
        JSON string with the extracted text content.
    """
    logger.info(f"Web fetch: {url}")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "\n\n... [content truncated]"

        result = json.dumps(
            {
                "url": url,
                "status_code": resp.status_code,
                "content_length": len(cleaned),
                "text": cleaned,
            }
        )
        logger.info(f"Web fetch returned {len(cleaned)} chars from {url}")
        return result

    except Exception as e:
        logger.error(f"Web fetch failed for {url}: {e}")
        return json.dumps({"error": str(e), "url": url})


def wait(seconds: int) -> str:
    """Pause execution for the specified number of seconds.

    Args:
        seconds: Number of seconds to wait (max 5400 = 90 minutes).

    Returns:
        Confirmation message.
    """
    max_wait = 5400  # 90 minutes
    seconds = min(seconds, max_wait)
    seconds = max(seconds, 1)

    logger.info(f"Waiting for {seconds} seconds ({seconds / 60:.1f} minutes)...")

    # We can't actually sleep here in the tool function because
    # the main loop needs to handle it. We return a special marker
    # that the main loop will detect and handle.
    return json.dumps(
        {
            "action": "wait",
            "seconds": seconds,
            "message": f"Initiating {seconds} second wait ({seconds / 60:.1f} minutes).",
        }
    )


def update_plan(new_plan: str) -> str:
    """Update the long-term PLAN document with new strategic musings.

    Args:
        new_plan: The new full content of the PLAN document.

    Returns:
        Confirmation message.
    """
    try:
        PLAN_FILE.write_text(new_plan, encoding="utf-8")
        logger.info(
            f"PLAN document updated ({len(new_plan)} chars, ~{len(new_plan.split())} words)"
        )
        return json.dumps(
            {"success": True, "message": "PLAN document updated successfully."}
        )
    except Exception as e:
        logger.error(f"Failed to update PLAN: {e}")
        return json.dumps({"success": False, "error": str(e)})


def _log_trade(
    action: str,
    contract_id: str,
    question: str,
    outcome: str,
    amount: float,
    limit_prob: Optional[float],
    shares: Optional[float],
    api_result: str,
    error_msg: str,
) -> None:
    """Append a trade record to trades.csv (human-readable log, not accessed by model)."""
    try:
        file_exists = TRADES_FILE.exists()
        with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "action", "contract_id", "question",
                    "outcome", "amount_mana", "limit_prob", "shares",
                    "api_result", "error",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                action,
                contract_id,
                question,
                outcome,
                f"{amount:.4f}",
                f"{limit_prob:.4f}" if limit_prob is not None else "",
                f"{shares:.4f}" if shares is not None else "",
                api_result,
                error_msg,
            ])
        logger.info(f"Trade logged to trades.csv: {action} {outcome} M${amount:.2f} on {contract_id}")
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")


def execute_tool(
    tool_name: str, tool_args: dict, manifold: ManifoldClient
) -> str:
    """Execute a tool call and return the result as a string.

    Args:
        tool_name: The name of the tool to execute.
        tool_args: The arguments for the tool.
        manifold: ManifoldClient instance.

    Returns:
        String result of the tool execution.
    """
    logger.info(f"Executing tool: {tool_name}({json.dumps(tool_args)})")

    try:
        if tool_name == "browse_markets":
            limit = tool_args.get("limit", 50)
            sort = tool_args.get("sort", "created-time")
            order = tool_args.get("order", "desc")
            result = manifold.browse_markets(limit=limit, sort=sort, order=order)
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "search_markets":
            term = tool_args.get("term", "")
            filter_type = tool_args.get("filter", "open")
            sort = tool_args.get("sort", "score")
            limit = tool_args.get("limit", 20)
            result = manifold.search_markets(
                term=term, filter_type=filter_type, sort=sort, limit=limit
            )
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "get_market":
            market_id = tool_args.get("market_id", "")
            result = manifold.get_market(market_id)
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "place_bet":
            contract_id = tool_args.get("contract_id", "")
            amount = float(tool_args.get("amount", 0))
            outcome = tool_args.get("outcome", "YES").upper()
            limit_prob = tool_args.get("limit_prob")

            # Try to get market question for the trade log
            question = ""
            try:
                market = manifold.get_market(contract_id)
                if "error" not in market:
                    question = market.get("question", "")
            except Exception:
                pass

            result = manifold.place_bet(
                contract_id=contract_id,
                amount=amount,
                outcome=outcome,
                limit_prob=limit_prob,
            )

            # Log trade
            api_ok = "error" not in result
            _log_trade(
                action="BUY",
                contract_id=contract_id,
                question=question,
                outcome=outcome,
                amount=amount,
                limit_prob=limit_prob,
                shares=result.get("shares") if api_ok else None,
                api_result="success" if api_ok else "failed",
                error_msg=result.get("error", "") if not api_ok else "",
            )

            return json.dumps(result, indent=2, default=str)

        elif tool_name == "sell_position":
            contract_id = tool_args.get("contract_id", "")
            shares = tool_args.get("shares")

            # Try to get market question for the trade log
            question = ""
            try:
                market = manifold.get_market(contract_id)
                if "error" not in market:
                    question = market.get("question", "")
            except Exception:
                pass

            result = manifold.sell_position(contract_id=contract_id, shares=shares)

            # Log trade
            api_ok = "error" not in result
            _log_trade(
                action="SELL",
                contract_id=contract_id,
                question=question,
                outcome="",
                amount=0,
                limit_prob=None,
                shares=shares if shares is not None else result.get("shares"),
                api_result="success" if api_ok else "failed",
                error_msg=result.get("error", "") if not api_ok else "",
            )

            return json.dumps(result, indent=2, default=str)

        elif tool_name == "get_positions":
            result = manifold.get_positions_summary()
            return json.dumps(result, indent=2, default=str)

        elif tool_name == "web_search":
            query = tool_args.get("query", "")
            max_results = tool_args.get("max_results", 5)
            return web_search(query=query, max_results=max_results)

        elif tool_name == "web_fetch":
            url = tool_args.get("url", "")
            max_chars = tool_args.get("max_chars", 8000)
            return web_fetch(url=url, max_chars=max_chars)

        elif tool_name == "wait":
            seconds = int(tool_args.get("seconds", 60))
            return wait(seconds=seconds)

        elif tool_name == "update_plan":
            new_plan = tool_args.get("new_plan", "")
            return update_plan(new_plan=new_plan)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error(f"Tool execution failed: {tool_name} - {e}")
        return json.dumps({"error": str(e), "tool": tool_name})


# === Tool Definitions for LLM ===

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "browse_markets",
            "description": "Browse recently created markets on Manifold. Returns markets ordered by creation date. Use search_markets to filter by status (open/closed) or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of markets to return (default: 50, max: 1000).",
                        "default": 50,
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort field: 'created-time', 'updated-time', 'last-bet-time', or 'last-comment-time'.",
                        "enum": ["created-time", "updated-time", "last-bet-time", "last-comment-time"],
                        "default": "created-time",
                    },
                    "order": {
                        "type": "string",
                        "description": "Sort order: 'asc' or 'desc'.",
                        "enum": ["asc", "desc"],
                        "default": "desc",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_markets",
            "description": "Search for specific markets on Manifold by keyword or term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Search term or keyword to find markets.",
                    },
                    "filter": {
                        "type": "string",
                        "description": "Filter: 'open', 'closed', 'resolved', or 'all'.",
                        "enum": ["open", "closed", "resolved", "all"],
                        "default": "open",
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort order: 'score', 'newest', 'close-date', 'most-popular', 'liquidity', '24-hour-vol'.",
                        "default": "score",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20).",
                        "default": 20,
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market",
            "description": "Get detailed information about a specific market by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The market/contract ID to fetch details for.",
                    },
                },
                "required": ["market_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_bet",
            "description": "Place a bet on a Manifold market. You can bet YES or NO. You can also set a limit probability for limit orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {
                        "type": "string",
                        "description": "The market/contract ID to bet on.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount of Mana (M$) to bet. Must be positive.",
                        "minimum": 0.01,
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Which side to bet on: 'YES' or 'NO'.",
                        "enum": ["YES", "NO"],
                    },
                    "limit_prob": {
                        "type": "number",
                        "description": "Optional: limit probability between 0 and 1 for a limit order. This is the worst probability you're willing to accept.",
                        "minimum": 0.01,
                        "maximum": 0.99,
                    },
                },
                "required": ["contract_id", "amount", "outcome"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sell_position",
            "description": "Sell your position (shares) in a Manifold market. If shares is not specified, sells all shares.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {
                        "type": "string",
                        "description": "The market/contract ID to sell your position in.",
                    },
                    "shares": {
                        "type": "number",
                        "description": "Optional: number of shares to sell. If omitted, sells all shares.",
                        "minimum": 0.01,
                    },
                },
                "required": ["contract_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": "Get your current Manifold account balance and all active positions in markets.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use this to research topics before placing bets. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and extract readable text content from a specific URL. Use this to read articles, news, or data sources in detail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of content to return (default: 8000).",
                        "default": 8000,
                        "minimum": 500,
                        "maximum": 20000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Pause execution for a specified time. Use when you have no good opportunities or want to wait for new information. Max 5400 seconds (90 minutes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to wait (minimum: 1, maximum: 5400 = 90 minutes).",
                        "minimum": 1,
                        "maximum": 5400,
                    },
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Update your long-term PLAN document. Use this to save strategic thoughts, observations, and plans that persist across context compaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_plan": {
                        "type": "string",
                        "description": "The new full content of the PLAN document. This replaces the existing plan entirely.",
                    },
                },
                "required": ["new_plan"],
            },
        },
    },
]
