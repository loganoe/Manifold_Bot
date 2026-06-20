#!/usr/bin/env python3
"""Test script: place a 1 mana bet then immediately sell to verify the API works."""

import json
import logging
import sys
from dotenv import load_dotenv
import os

from manifold import ManifoldClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test")

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

# 2. Find a far-away binary market
logger.info("Finding a far-away binary market...")
markets = client.browse_markets(limit=100, sort="created-time", order="desc")
if isinstance(markets, dict) and "error" in markets:
    logger.error(f"Browse failed: {markets['error']}")
    sys.exit(1)

# Find a binary market closing far in the future with some volume
target = None
for m in markets:
    if not isinstance(m, dict):
        continue
    if m.get("outcomeType") != "BINARY":
        continue
    if m.get("isResolved", False):
        continue
    close_time = m.get("closeTime", 0)
    # Must be at least 2 months away (far-away)
    from time import time
    two_months_ms = 60 * 24 * 60 * 60 * 1000
    if close_time > (time() * 1000) + two_months_ms:
        target = m
        break

if target is None:
    # Fallback: just take any binary
    for m in markets:
        if isinstance(m, dict) and m.get("outcomeType") == "BINARY" and not m.get("isResolved", False):
            target = m
            break

if target is None:
    logger.error("No suitable binary market found!")
    sys.exit(1)

market_id = target["id"]
question = target["question"]
prob = target.get("probability", 0.5)
close_ms = target.get("closeTime", 0)
from datetime import datetime, timezone
close_dt = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc)
logger.info(f"Selected: '{question}' (id={market_id}, prob={prob:.2%}, closes {close_dt.date()})")

# 3. Place a 1 mana bet
bet_amount = 1.0
outcome = "YES"
logger.info(f"Placing bet: M${bet_amount} {outcome} on {market_id}...")
bet_result = client.place_bet(contract_id=market_id, amount=bet_amount, outcome=outcome)

if "error" in bet_result:
    logger.error(f"BET FAILED: {bet_result['error']}")
    logger.error("Full response: " + json.dumps(bet_result, indent=2, default=str))
    sys.exit(1)

logger.info(f"BET SUCCESS! Shares bought: {bet_result.get('shares', 'N/A')}")
logger.info(f"Bet details: {json.dumps(bet_result, indent=2, default=str)[:500]}")

# 4. Check balance after bet
me_after_bet = client.get_me()
balance_after_bet = me_after_bet.get("balance", 0)
logger.info(f"Balance after bet: M${balance_after_bet}")

# 5. Immediately sell
logger.info(f"Selling position in {market_id} (outcome={outcome})...")
sell_result = client.sell_position(contract_id=market_id, outcome=outcome)

if "error" in sell_result:
    logger.error(f"SELL FAILED: {sell_result['error']}")
    logger.error("Full response: " + json.dumps(sell_result, indent=2, default=str))
    sys.exit(1)

logger.info(f"SELL SUCCESS!")
logger.info(f"Sell details: {json.dumps(sell_result, indent=2, default=str)[:500]}")

# 6. Final balance check
me_final = client.get_me()
balance_final = me_final.get("balance", 0)
logger.info(f"Final balance: M${balance_final}")
logger.info(f"Net change: M${balance_final - balance_before:.2f}")

logger.info("=" * 60)
logger.info("TEST PASSED: Both betting and selling work correctly!")
logger.info("=" * 60)
