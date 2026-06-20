#!/usr/bin/env python3
"""Test script: verify reasoning traces work with the primary model."""

import json
import logging
import sys
from dotenv import load_dotenv
import os

from llm import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test-reasoning")

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
if not api_key:
    logger.error("OPENROUTER_API_KEY not found in .env")
    sys.exit(1)

client = LLMClient(api_key)
assert client.current_model == "nvidia/nemotron-3-ultra-550b-a55b:free", \
    f"Expected primary model, got {client.current_model}"

# Simple trading-like prompt with tools to trigger reasoning
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful trading bot. Before answering, think through "
            "your reasoning carefully step by step."
        ),
    },
    {
        "role": "user",
        "content": (
            "What is 2+2? Use browse_markets as your first action to see what's available. "
            "This is a test — just browse and then explain what you found."
        ),
    },
]

# Minimal tool definitions matching the bot's format
test_tools = [
    {
        "type": "function",
        "function": {
            "name": "browse_markets",
            "description": "Browse markets",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

logger.info("=" * 60)
logger.info("Testing reasoning with nvidia/nemotron-3-ultra-550b-a55b:free")
logger.info("=" * 60)

response = client.chat(messages, test_tools)

if response is None:
    logger.error("FAILED: No response from model.")
    sys.exit(1)

# Check for reasoning in raw response
choices = response.get("choices", [])
message = choices[0].get("message", {})
reasoning_raw = message.get("reasoning", "")
content = message.get("content", "")
tool_calls = message.get("tool_calls", [])

logger.info(f"Content present: {'Yes' if content else 'No'} ({len(content or '')} chars)")
logger.info(f"Tool calls: {len(tool_calls)}")
logger.info(f"Reasoning in response: {'Yes' if reasoning_raw else 'No'} ({len(reasoning_raw)} chars)")

# Show snippet of reasoning
if reasoning_raw:
    logger.info("=" * 60)
    logger.info("REASONING DETECTED — it works!")
    logger.info(f"First 300 chars of reasoning: {reasoning_raw[:300]}...")
    logger.info("=" * 60)
else:
    logger.warning("WARNING: No reasoning trace found in response!")
    logger.warning("Full response keys: " + str(list(message.keys())))
    # Check if reasoning is nested somewhere else
    logger.warning("Response structure: " + json.dumps({
        k: type(v).__name__ + (f"({len(v)} chars)" if isinstance(v, str) else "")
        for k, v in message.items()
    }, indent=2))

# Also verify the extract_reasoning method works
reasoning_from_extractor = client.extract_reasoning(response)
if reasoning_from_extractor:
    logger.info(f"extract_reasoning() returned {len(reasoning_from_extractor)} chars — logging works")
else:
    logger.warning("extract_reasoning() returned empty — logging won't capture reasoning")

logger.info("\nTest complete.")
if reasoning_raw:
    logger.info("✅ PASS: Reasoning traces are working with the primary model.")
    sys.exit(0)
else:
    logger.error("❌ FAIL: No reasoning trace detected.")
    sys.exit(1)
