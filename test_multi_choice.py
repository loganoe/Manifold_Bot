#!/usr/bin/env python3
"""Test multi-choice market betting and selling."""

import json
import logging
import sys
from dotenv import load_dotenv
import os

from manifold import ManifoldClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test-multi")

load_dotenv()
api_key = os.getenv("MANIFOLD_API_KEY", "").strip()
if not api_key:
    logger.error("MANIFOLD_API_KEY not found in .env")
    sys.exit(1)

client = ManifoldClient(api_key)

# 1. Verify connection
me = client.get_me()
if "error" in me:
    logger.error(f"Connection failed: {me['error']}")
    sys.exit(1)
balance_before = me.get("balance", 0)
logger.info(f"Connected! Balance: M${balance_before}")

# 2. Find a multi-choice market
logger.info("Finding a MULTIPLE_CHOICE market...")
markets = client.browse_markets(limit=100, sort="created-time", order="desc")
if isinstance(markets, dict) and "error" in markets:
    logger.error(f"Browse failed: {markets['error']}")
    sys.exit(1)

target = None
for m in markets:
    if not isinstance(m, dict):
        continue
    if m.get("outcomeType") == "MULTIPLE_CHOICE" and not m.get("isResolved", False):
        target = m
        break

if target is None:
    logger.error("No MULTIPLE_CHOICE market found!")
    sys.exit(1)

market_id = target["id"]
question = target["question"]
logger.info(f"Selected: '{question}' (id={market_id})")

# 3. Get full market details to see answers
logger.info(f"Getting full market details for {market_id}...")
market = client.get_market(market_id)
if "error" in market:
    logger.error(f"get_market failed: {market['error']}")
    sys.exit(1)

answers = market.get("answers", [])
logger.info(f"Market has {len(answers)} answers:")
for a in answers:
    logger.info(f"  - {a.get('id')}: '{a.get('text')}' (prob: {a.get('probability', '?')})")

if not answers:
    logger.error("No answers found!")
    sys.exit(1)

# Pick the first answer
answer = answers[0]
answer_id = answer["id"]
answer_text = answer["text"]
answer_prob = answer.get("probability", 0)
logger.info(f"Betting on: '{answer_text}' (id={answer_id}, prob={answer_prob})")

# 4. Place a 1 mana bet
bet_amount = 1.0
logger.info(f"Placing bet: M${bet_amount} YES on {market_id} answer_id={answer_id}...")
bet_result = client.place_bet(
    contract_id=market_id,
    amount=bet_amount,
    outcome="YES",
    answer_id=answer_id,
)

if "error" in bet_result:
    logger.error(f"BET FAILED: {bet_result['error']}")
    logger.error(json.dumps(bet_result, indent=2, default=str))
    sys.exit(1)

logger.info(f"BET SUCCESS! Shares: {bet_result.get('shares', 'N/A')}")
logger.info(f"Bet details: {json.dumps(bet_result, indent=2, default=str)[:400]}")

me_after = client.get_me()
logger.info(f"Balance after bet: M${me_after.get('balance', '?')}")

# 5. Immediately sell
logger.info(f"Selling position in {market_id} answer_id={answer_id}...")
sell_result = client.sell_position(
    contract_id=market_id,
    outcome="YES",
    answer_id=answer_id,
)

if "error" in sell_result:
    logger.error(f"SELL FAILED: {sell_result['error']}")
    logger.error(json.dumps(sell_result, indent=2, default=str))
    sys.exit(1)

logger.info(f"SELL SUCCESS!")
logger.info(f"Sell details: {json.dumps(sell_result, indent=2, default=str)[:400]}")

me_final = client.get_me()
logger.info(f"Final balance: M${me_final.get('balance', '?')}")
logger.info(f"Net change: M${me_final.get('balance', 0) - balance_before:.2f}")

logger.info("=" * 60)
logger.info("✅ PASS: Multi-choice betting and selling work correctly!")
logger.info("=" * 60)
