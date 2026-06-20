"""
Manifold Markets API client.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MANIFOLD_BASE = "https://api.manifold.markets/v0"


class ManifoldClient:
    """Client for the Manifold Markets API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            }
        )

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make a GET request to the Manifold API."""
        url = f"{MANIFOLD_BASE}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Manifold GET {path} failed: {e}")
            return {"error": str(e)}

    def _post(self, path: str, data: dict) -> dict:
        """Make a POST request to the Manifold API."""
        url = f"{MANIFOLD_BASE}{path}"
        try:
            resp = self.session.post(url, json=data, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Manifold POST {path} failed: {e}")
            return {"error": str(e)}

    def browse_markets(self, limit: int = 50, sort: str = "created-time", order: str = "desc") -> dict:
        """Browse recently created markets. Does NOT support 'filter' — use search_markets for filtering by status."""
        return self._get("/markets", {"limit": limit, "sort": sort, "order": order})

    def search_markets(
        self,
        term: str,
        filter_type: str = "open",
        sort: str = "score",
        limit: int = 20,
    ) -> dict:
        """Search for markets by keyword."""
        return self._get(
            "/search-markets",
            {"term": term, "filter": filter_type, "sort": sort, "limit": limit},
        )

    def get_market(self, market_id: str) -> dict:
        """Get full details for a specific market."""
        return self._get(f"/market/{market_id}")

    def place_bet(
        self,
        contract_id: str,
        amount: float,
        outcome: str,
        limit_prob: Optional[float] = None,
        answer_id: Optional[str] = None,
    ) -> dict:
        """Place a bet on a market.

        Args:
            contract_id: The market/contract ID.
            amount: Amount of Mana to bet.
            outcome: "YES" or "NO".
            limit_prob: Optional limit probability (0-1) for a limit order.
            answer_id: Required for MULTIPLE_CHOICE markets — the answer ID to bet on.
        """
        data = {
            "contractId": contract_id,
            "amount": amount,
            "outcome": outcome,
        }
        if limit_prob is not None:
            data["limitProb"] = limit_prob
        if answer_id is not None:
            data["answerId"] = answer_id
        return self._post("/bet", data)

    def sell_position(self, contract_id: str, outcome: str, shares: Optional[float] = None, answer_id: Optional[str] = None) -> dict:
        """Sell all or some shares in a market.

        Args:
            contract_id: The market/contract ID.
            outcome: Which outcome to sell ("YES" or "NO"). Required by the API.
            shares: Optional number of shares to sell. If None, sells all.
            answer_id: Required for MULTIPLE_CHOICE markets — the answer ID to sell.
        """
        data: dict = {"outcome": outcome}
        if shares is not None:
            data["shares"] = shares
        if answer_id is not None:
            data["answerId"] = answer_id
        return self._post(f"/market/{contract_id}/sell", data)

    def get_me(self) -> dict:
        """Get current user info including balance."""
        return self._get("/me")

    def get_bets(
        self, limit: int = 100, order: str = "desc"
    ) -> list[dict]:
        """Get recent bets for the authenticated user."""
        return self._get("/bets", {"limit": limit, "order": order})

    def get_positions_summary(self) -> dict:
        """Get a summary of current positions and balance."""
        me = self.get_me()
        if "error" in me:
            return me

        bets = self.get_bets(limit=500)
        if isinstance(bets, dict) and "error" in bets:
            bets = []
        elif isinstance(bets, list) and len(bets) >= 500:
            logger.warning(
                "get_positions_summary: max 200 bets fetched; "
                "some positions may be missing"
            )

        # Build positions from bets
        positions = {}
        if isinstance(bets, list):
            for bet in bets:
                contract_id = bet.get("contractId", "")
                answer_id = bet.get("answerId", "")
                # For multi-choice markets, use contractId+answerId as key
                pos_key = f"{contract_id}:{answer_id}" if answer_id else contract_id
                if pos_key not in positions:
                    positions[pos_key] = {
                        "contractId": contract_id,
                        "question": bet.get("contractQuestion", bet.get("question", "Unknown")),
                        "slug": bet.get("contractSlug", ""),
                        "outcome": bet.get("outcome", ""),
                        "shares": 0,
                        "total_invested": 0,
                    }
                    if answer_id:
                        positions[pos_key]["answerId"] = answer_id
                        positions[pos_key]["answerText"] = bet.get("answerText", bet.get("answer", "")) or ""
                # Accumulate shares
                shares = bet.get("shares", 0)
                amount = bet.get("amount", 0)
                if bet.get("isSold", False) or not bet.get("isFilled", True):
                    continue
                if bet.get("isAnte", False):
                    # Ante bets shouldn't count
                    continue
                if shares > 0:
                    positions[pos_key]["shares"] += shares
                    positions[pos_key]["total_invested"] += amount

        # Only include positions with non-zero shares
        active_positions = [
            p for p in positions.values() if abs(p["shares"]) > 0.001
        ]

        return {
            "balance": me.get("balance", 0),
            "totalDeposits": me.get("totalDeposits", 0),
            "profitCached": me.get("profitCached", {}),
            "positions": active_positions,
            "num_positions": len(active_positions),
        }
