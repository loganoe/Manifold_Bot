#!/usr/bin/env python3
"""
Manifold Bot - An LLM-powered trading bot on Manifold Markets via OpenRouter.

Usage:
    python main.py

Requires a .env file with MANIFOLD_API_KEY and OPENROUTER_API_KEY.
"""

import json
import logging
import os
import signal
import sys
import time as time_module
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from llm import LLMClient
from manifold import ManifoldClient
from tools import TOOL_DEFINITIONS, execute_tool

# === Configuration ===
load_dotenv()

MANIFOLD_API_KEY = os.getenv("MANIFOLD_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

PLAN_FILE = Path("PLAN.md")
STATE_FILE = Path("state.json")

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")

# Graceful shutdown: first SIGINT saves state, second force-exits
shutdown_requested = False
_signal_count = 0
_messages_ref = None
_iteration_ref = None
_llm_ref = None


def handle_signal(signum, frame):
    global shutdown_requested, _signal_count
    _signal_count += 1
    sig_name = signal.Signals(signum).name

    if _signal_count == 1:
        logger.info(f"Received {sig_name}, shutting down gracefully... (press Ctrl+C again to force exit)")
        shutdown_requested = True
        # Save state immediately from the signal handler
        if _messages_ref is not None and _llm_ref is not None:
            try:
                save_state(_messages_ref, _iteration_ref or 0, _llm_ref.current_model)
                logger.info("State saved from signal handler.")
            except Exception as e:
                logger.error(f"Signal-handler state save failed: {e}")
    else:
        logger.warning(f"Received second {sig_name}, forcing immediate exit...")
        os._exit(0)


def _do_shutdown_save(messages: list, iteration: int, llm: LLMClient) -> None:
    """Save state before shutdown (called from signal context via flag)."""
    try:
        save_state(messages, iteration, llm.current_model)
    except Exception as e:
        logger.error(f"Failed to save state during shutdown: {e}")


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# === System Prompt ===

SYSTEM_PROMPT = """You are an autonomous trading bot on Manifold Markets. Your sole objective is to earn as much Mana (M$) as possible through strategic betting on prediction markets.

The current date and time is: {current_date}

## Core Principles

1. **RESEARCH FIRST — THIS IS CRITICAL**: You MUST research before EVERY bet. Use web_search to find recent news, data, and expert opinions relevant to the market. Then use web_fetch to read a primary source article in full. Only then should you consider betting. Betting without research is gambling, and gambling loses money. You are a RESEARCHER first, a trader second.

2. **RESEARCH WORKFLOW**: For every market you consider betting on: (a) web_search the topic with 2-3 different queries to get broad context, (b) web_fetch at least one authoritative article (news site, official data source, expert analysis), (c) form a probability estimate based on the evidence, (d) compare your estimate to the market price. If the gap is large enough (>5-10 percentage points), place a bet.

3. **FIND EDGES**: Look for markets where the current probability appears mispriced based on your research. A market at 20% for an event your research suggests is 40% likely is a good opportunity. The bigger the research gap, the bigger your edge.

4. **DIVERSIFY**: Never put all your Mana into a single bet. Spread your capital across multiple uncorrelated markets. A good rule: no single bet should exceed 20% of your balance.

5. **RISK MANAGEMENT**: Start with smaller bets to test your strategy. Scale up only when you have a track record of success. Don't chase losses.

6. **PATIENCE**: If you don't see compelling opportunities or just want to wait to get more information, use the wait() tool to pause. It's better to do nothing than to make bad bets. The maximum wait is 5400 seconds (90 minutes).

7. **MONITOR POSITIONS**: Regularly check your positions. Sell positions that have moved in your favor to lock in profits, or cut losses if the fundamentals have changed against you.

8. **LEARN AND ADAPT**: Use the update_plan() tool to record your observations, strategies, and lessons learned. This plan persists across context compaction and helps you maintain a coherent long-term strategy.

9. **THINK PROBABILISTICALLY**: Manifold uses binary options. Your job is to estimate true probabilities. A market at 70% means YES shares cost M$0.70 each. If you think the true probability is 85%, buying YES at 70% has positive expected value.

10. **AVOID LONG-TERM PREDICTION**: Markets that resolve far in the future (more than 3 months away) are not only harder to predict, but because they resolve farther in the future, they will add to your existing balance less quickly and thus restrict compounding growth. Focus on markets that resolve soon, preferably <30 days away, but at most 3 months away. 

## Market Mechanics

- Manifold markets are binary: YES or NO. 
- YES shares pay M$1 if the event occurs, M$0 if not.
- NO shares pay M$1 if the event does NOT occur.
- Current probability = current price of YES shares.
- You can place market orders (immediate) or limit orders (specify worst acceptable probability).
- You can sell your positions at any time.

## Your Tools

You have access to the following tools. Use them strategically:

- **browse_markets**: See what markets are available recently.
- **search_markets**: Find specific markets by keyword.
- **get_market**: Get detailed info about a market (rules, close date, current stats).
- **place_bet**: Bet on a market (YES or NO, with optional limit probability).
- **sell_position**: Exit a position (sell all or some shares).
- **get_positions**: View your balance and current positions.
- **web_search**: Research topics using web search. ALWAYS use this before betting.
- **web_fetch**: Read full articles/webpages for deeper research.
- **wait**: Pause (up to 90 minutes) when no good opportunities exist. When pausing, specifiy beforehand a condition for whether you will go back to waiting after you unpause (pausing multiple times in a row if no opportunities are found is perfectly acceptable). 
- **update_plan**: Save your strategic thoughts to your persistent PLAN document. 

## Current Status

Your current account balance and positions are shown below. Your PLAN document follows after that. Use both to inform your decisions.

---

{positions_info}

---

## PLAN Document

{plan_content}

---

Now take action. What will you do next? Remember: RESEARCH before you bet."""


def load_plan() -> str:
    """Load the PLAN document from disk, creating it if needed."""
    if PLAN_FILE.exists():
        return PLAN_FILE.read_text(encoding="utf-8")
    else:
        default_plan = (
            "# Trading Bot Strategic Plan\n\n"
            "## Goals\n"
            "- Earn as much Mana as possible through strategic betting\n\n"
            "## Strategy Notes\n"
            "- No strategy defined yet. Start by browsing markets and doing research.\n\n"
            "## Observations\n"
            "- Bot just started. Need to explore available markets.\n\n"
            "## Lessons Learned\n"
            "- No lessons yet.\n"
        )
        PLAN_FILE.write_text(default_plan, encoding="utf-8")
        return default_plan


def save_state(messages: list, iteration: int, current_model: str) -> None:
    """Save conversation state to disk for later resume."""
    try:
        state = {
            "messages": messages,
            "iteration": iteration,
            "current_model": current_model,
        }
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        logger.info(f"State saved to {STATE_FILE} ({len(messages)} messages, iteration {iteration})")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_state() -> dict | None:
    """Load saved conversation state from disk, or None if not found."""
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        logger.info(
            f"Loaded state from {STATE_FILE}: "
            f"{len(state.get('messages', []))} messages, "
            f"iteration {state.get('iteration', 0)}"
        )
        return state
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return None


def build_system_message(manifold: ManifoldClient) -> dict:
    """Build the system message with current positions and PLAN."""
    # Get current positions
    positions_data = manifold.get_positions_summary()
    if isinstance(positions_data, dict) and "error" in positions_data:
        positions_str = f"Error fetching positions: {positions_data['error']}"
    else:
        positions_str = json.dumps(positions_data, indent=2, default=str)

    # Load PLAN
    plan_content = load_plan()

    # Build system message (use replace to avoid str.format() issues
    # with JSON curly braces in positions data)
    current_date = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
    content = (
        SYSTEM_PROMPT
        .replace("{current_date}", current_date)
        .replace("{positions_info}", positions_str)
        .replace("{plan_content}", plan_content)
    )

    return {"role": "system", "content": content}


def main():
    """Main entry point for the trading bot."""
    logger.info("=" * 60)
    logger.info("Manifold Trading Bot starting...")
    logger.info("=" * 60)

    # Validate API keys
    if not MANIFOLD_API_KEY:
        logger.error(
            "MANIFOLD_API_KEY not found in .env file. "
            "Copy .env.example to .env and add your keys."
        )
        sys.exit(1)
    if not OPENROUTER_API_KEY:
        logger.error(
            "OPENROUTER_API_KEY not found in .env file. "
            "Copy .env.example to .env and add your keys."
        )
        sys.exit(1)

    # Initialize clients
    manifold = ManifoldClient(MANIFOLD_API_KEY)
    llm = LLMClient(OPENROUTER_API_KEY)

    # Verify API connectivity
    logger.info("Verifying Manifold API connectivity...")
    me = manifold.get_me()
    if "error" in me:
        logger.error(f"Manifold API connection failed: {me['error']}")
        logger.error("Please check your MANIFOLD_API_KEY.")
        sys.exit(1)
    logger.info(
        f"Connected to Manifold as user. Balance: M${me.get('balance', 'N/A')}"
    )

    # Load saved state or start fresh
    fresh_start = "--fresh" in sys.argv
    saved_state = None if fresh_start else load_state()

    if saved_state and saved_state.get("messages"):
        messages: list = saved_state["messages"]
        iteration = saved_state.get("iteration", 0)
        saved_model = saved_state.get("current_model", "")
        if saved_model:
            llm.current_model = saved_model
        # Refresh the system message with latest positions
        messages[0] = build_system_message(manifold)
        logger.info(f"Resumed from saved state at iteration {iteration}")
    else:
        if fresh_start:
            logger.info("--fresh flag: starting from scratch")
            STATE_FILE.unlink(missing_ok=True)
        else:
            logger.info("No saved state found, starting fresh")

        messages = [build_system_message(manifold)]
        messages.append(
            {
                "role": "user",
                "content": (
                    "You are now active. Review your balance and current positions "
                    "above. Browse available markets, research opportunities, and "
                    "take action to earn Mana. What will you do?"
                ),
            }
        )
        iteration = 0

    # Main loop — update refs so signal handler can access state for emergency saves
    global _messages_ref, _iteration_ref, _llm_ref
    _llm_ref = llm

    consecutive_errors = 0
    max_consecutive_errors = 5

    # Main loop
    while not shutdown_requested:
        _messages_ref = messages
        _iteration_ref = iteration
        iteration += 1
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'=' * 60}")

        try:
            # Refresh system message with latest positions and PLAN
            messages[0] = build_system_message(manifold)

            # Compensate context if needed
            plan_content = load_plan()
            messages = llm.compact_context(messages, plan_content)

            # Send to LLM
            logger.info(
                f"Sending {len(messages)} messages to LLM "
                f"(model: {llm.current_model})..."
            )

            response = llm.chat(messages, TOOL_DEFINITIONS)

            if response is None:
                logger.error("LLM call failed - all models unavailable.")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"{max_consecutive_errors} consecutive LLM failures. "
                        "Exiting."
                    )
                    break
                backoff = min(30, 2**consecutive_errors)
                logger.info(f"Backing off for {backoff}s...")
                time_module.sleep(backoff)
                continue

            consecutive_errors = 0

            # Process response
            tool_calls = llm.extract_tool_calls(response)
            text_response = llm.extract_text_response(response)

            choice = response.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason", "")
            message = choice.get("message", {})

            # Log usage if available
            usage = response.get("usage", {})
            if usage:
                logger.info(
                    f"LLM usage: {usage.get('prompt_tokens', '?')} prompt + "
                    f"{usage.get('completion_tokens', '?')} completion = "
                    f"{usage.get('total_tokens', '?')} total tokens"
                )

            # Add assistant message to conversation
            assistant_msg = {"role": "assistant", "content": text_response or None}

            if tool_calls:
                # Add tool calls to assistant message in OpenAI format
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ]

            messages.append(assistant_msg)

            # Execute tool calls
            if tool_calls:
                logger.info(
                    f"Executing {len(tool_calls)} tool call(s): "
                    f"{[tc['name'] for tc in tool_calls]}"
                )

                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]

                    result = execute_tool(tool_name, tool_args, manifold)

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tool_name,
                            "content": result,
                        }
                    )

                    # Handle wait() - actually sleep
                    if tool_name == "wait":
                        try:
                            wait_data = json.loads(result)
                            if wait_data.get("action") == "wait":
                                seconds = wait_data["seconds"]
                                logger.info(
                                    f"WAIT: Sleeping for {seconds}s "
                                    f"({seconds/60:.1f}m)..."
                                )

                                # Sleep in chunks to allow graceful shutdown
                                chunk = 10
                                elapsed = 0
                                while elapsed < seconds and not shutdown_requested:
                                    sleep_time = min(chunk, seconds - elapsed)
                                    time_module.sleep(sleep_time)
                                    elapsed += sleep_time

                                # After waking, add a message to restart the loop
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            f"WAIT COMPLETE: You waited "
                                            f"approximately {elapsed} seconds "
                                            f"({elapsed/60:.1f} minutes). "
                                            f"What would you like to do now?"
                                        ),
                                    }
                                )
                                logger.info("Wait complete.")
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Failed to parse wait response: {e}")

                    # Handle update_plan - reload plan for next iteration
                    if tool_name == "update_plan":
                        logger.info("PLAN updated, will refresh system message.")

                    # Small delay between tool executions to respect rate limits
                    time_module.sleep(0.5)

            elif finish_reason == "stop":
                # LLM responded with text only (no tool calls)
                logger.info(
                    f"LLM text response: {text_response[:200]}..."
                )
                # Ask the LLM to continue
                messages.append(
                    {
                        "role": "user",
                        "content": "Continue. What will you do next?",
                    }
                )
            else:
                logger.warning(
                    f"Unexpected finish reason: {finish_reason}"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Continue. What will you do next?",
                    }
                )

            # Small delay between iterations
            time_module.sleep(1)

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(
                    f"{max_consecutive_errors} consecutive errors. Exiting."
                )
                break
            backoff = min(30, 2**consecutive_errors)
            logger.info(f"Backing off for {backoff}s...")
            time_module.sleep(backoff)

    logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    main()
