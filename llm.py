"""
OpenRouter LLM client with model fallback and context management.
"""

import json
import logging
import time as time_module
from typing import Any, Optional, Union

import requests
import tiktoken

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "poolside/laguna-m.1:free"
MAX_TIMEOUTS_BEFORE_FALLBACK = 3
TIMEOUT_SECONDS = 120
MAX_CONTEXT_TOKENS = 100_000
RECENT_MESSAGES_TO_KEEP = 15

# Tokenizer for counting
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text))


def count_message_tokens(message: dict) -> int:
    """Count tokens in a single message dict (approximate)."""
    total = 4  # message framing overhead
    for key, value in message.items():
        if value is None:
            continue
        if key == "tool_calls":
            for tc in value:
                if "function" in tc:
                    total += count_tokens(
                        json.dumps(tc["function"].get("arguments", ""))
                    )
                    total += count_tokens(tc["function"].get("name", ""))
        elif key == "tool_call_id":
            total += count_tokens(str(value))
        elif isinstance(value, str):
            total += count_tokens(value)
        elif isinstance(value, list):
            total += count_tokens(json.dumps(value, default=str))
        elif isinstance(value, dict):
            total += count_tokens(json.dumps(value, default=str))
        else:
            total += count_tokens(str(value))
    return total


def count_conversation_tokens(messages: list[dict]) -> int:
    """Count total tokens in a conversation."""
    total = 0
    for msg in messages:
        total += count_message_tokens(msg)
    return total


class LLMClient:
    """OpenRouter LLM client with model fallback and context compaction."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_model = PRIMARY_MODEL
        self.consecutive_timeouts = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/manifold-bot",
                "X-Title": "Manifold Trading Bot",
            }
        )

    def _call_api(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """Make a single API call to OpenRouter."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = self.session.post(
                f"{OPENROUTER_BASE}/chat/completions",
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            if resp.status_code == 408 or resp.status_code == 504:
                # Timeout
                logger.warning(
                    f"Timeout from {model} (status {resp.status_code})"
                )
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout from {model}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error to {model}: {e}")
            return None
        except Exception as e:
            logger.error(f"API call to {model} failed: {e}")
            return None

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """Send a chat completion request with automatic model fallback.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of tool definitions.

        Returns:
            The API response dict, or None if all models fail.
        """
        # Try primary model first
        model = self.current_model
        logger.info(
            f"Calling {model} with {len(messages)} messages "
            f"(~{sum(count_message_tokens(m) for m in messages)} tokens)"
        )

        response = self._call_api(model, messages, tools)

        if response is not None:
            self.consecutive_timeouts = 0
            # Check if we should try switching back to primary
            if model == FALLBACK_MODEL:
                logger.info(
                    "Considering switching back to primary model..."
                )
                # Simple ping to check if primary model is responsive
                probe = self._call_api(
                    PRIMARY_MODEL,
                    [{"role": "user", "content": "ping"}],
                    tools=None,
                )
                if probe is not None:
                    self.current_model = PRIMARY_MODEL
                    logger.info("Switched back to primary model.")

            return response

        # Primary timed out
        self.consecutive_timeouts += 1
        logger.warning(
            f"Primary model timeout ({self.consecutive_timeouts}/{MAX_TIMEOUTS_BEFORE_FALLBACK})"
        )

        if self.consecutive_timeouts >= MAX_TIMEOUTS_BEFORE_FALLBACK:
            logger.warning(
                f"Falling back to {FALLBACK_MODEL} after {self.consecutive_timeouts} timeouts."
            )
            self.current_model = FALLBACK_MODEL
            response = self._call_api(FALLBACK_MODEL, messages, tools)
            if response is not None:
                self.consecutive_timeouts = 0
                return response

            logger.error("Fallback model also failed!")
            return None

        # Retry primary one more time with backoff
        time_module.sleep(2)
        response = self._call_api(model, messages, tools)
        if response is not None:
            self.consecutive_timeouts = 0
            return response

        logger.error("Primary model failed after retry.")
        return None

    def extract_tool_calls(
        self, response: dict
    ) -> list[dict]:
        """Extract tool calls from an API response.

        Returns a list of dicts with 'id', 'name', and 'arguments' keys.
        """
        choices = response.get("choices", [])
        if not choices:
            return []

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        result = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}

            result.append(
                {
                    "id": tc.get("id", ""),
                    "name": name,
                    "arguments": arguments,
                }
            )

        return result

    def extract_text_response(self, response: dict) -> str:
        """Extract text content from a non-tool-call response."""
        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def compact_context(
        self, messages: list[dict], plan_content: str
    ) -> list[dict]:
        """Compact conversation history when exceeding token limit.

        Keeps the system message, PLAN, recent messages verbatim,
        and summarizes older messages.

        Args:
            messages: Full message list including system message.
            plan_content: The PLAN document content.

        Returns:
            Compacted message list.
        """
        total_tokens = count_conversation_tokens(messages)
        logger.info(
            f"Context: {total_tokens} tokens across {len(messages)} messages"
        )

        if total_tokens < MAX_CONTEXT_TOKENS * 0.8:
            return messages  # No compaction needed

        logger.warning(
            f"Compacting context ({total_tokens} tokens > "
            f"{MAX_CONTEXT_TOKENS * 0.8} threshold)"
        )

        # Build compaction prompt
        # Summarize everything except system message + last N messages
        system_msg = messages[0] if messages else {"role": "system", "content": ""}
        recent_msgs = messages[-RECENT_MESSAGES_TO_KEEP:]
        old_msgs = messages[1:-RECENT_MESSAGES_TO_KEEP]

        if not old_msgs:
            return messages

        # Format old messages for summarization
        old_text = ""
        for msg in old_msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    old_text += f"\n[Assistant called tool: {func.get('name', '')} with args: {func.get('arguments', '')}]\n"

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                old_text += (
                    f"\n[Tool result for {tool_id}]: "
                    f"{content[:500]}...\n"
                )
            elif content:
                old_text += f"\n[{role}]: {content[:1000]}\n"

        if len(old_text) > 10000:
            old_text = old_text[:10000] + "\n... [further history truncated for summarization]\n"

        summary_prompt = f"""Summarize the following conversation history between an AI trading bot and its tool results. Focus on:
1. What markets were examined and the findings
2. What bets were placed and why
3. What research was done
4. Key strategic decisions made
5. What the bot learned

For context, here is the bot's current strategic PLAN:
{plan_content[:3000]}

Conversation history to summarize:
{old_text}

Produce a concise summary (under 1000 words)."""

        # Use the LLM to summarize (use minimal call)
        logger.info("Calling LLM for context compaction...")
        summary_response = self._call_api(
            self.current_model,
            [{"role": "user", "content": summary_prompt}],
            tools=None,
        )

        if summary_response is None:
            logger.error("Compaction LLM call failed, using truncation fallback")
            # Fall back to simple truncation: keep system + recent messages
            return [system_msg] + recent_msgs

        summary_text = self.extract_text_response(summary_response)
        if not summary_text or len(summary_text) < 50:
            logger.warning("Compaction produced insufficient summary, using truncation")
            return [system_msg] + recent_msgs

        # Rebuild messages: system + summary + recent
        compacted = [
            system_msg,
            {
                "role": "user",
                "content": (
                    "[CONTEXT COMPACTION: The following is a summary of "
                    "previous conversation history.]\n\n"
                    f"{summary_text}\n\n"
                    "[End of compacted context. Recent messages follow.]"
                ),
            },
        ] + recent_msgs

        new_tokens = count_conversation_tokens(compacted)
        logger.info(
            f"Compacted: {total_tokens} → {new_tokens} tokens "
            f"({len(messages)} → {len(compacted)} messages)"
        )
        return compacted
