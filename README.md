# Manifold Trading Bot

An LLM-powered autonomous trading bot that earns Mana (M$) on [Manifold Markets](https://manifold.markets) using AI models via [OpenRouter](https://openrouter.ai).

## How It Works

The bot runs an infinite loop where an LLM (default: NVIDIA Nemotron, with automatic fallback to Poolside Laguna) decides what actions to take. It has access to:

- **Market Tools**: Browse available markets, search for specific markets, get market details, place bets (market and limit orders), sell positions, check balance
- **Research Tools**: Web search (via DuckDuckGo, free & unlimited) and web page fetching
- **Strategy Tools**: A persistent PLAN document for long-term strategic musings, and a wait() function to pause when no opportunities exist
- **Context Management**: Automatic conversation compaction when the context window approaches 100K tokens, preserving all critical information

## Setup

### 1. Get API Keys

- **Manifold Markets**: Go to [your profile](https://manifold.markets/profile) and generate an API key
- **OpenRouter**: Go to [OpenRouter Keys](https://openrouter.ai/keys) and create an API key (free tier: 50 requests/day, or 1,000/day with $10+ credits)

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your actual API keys
nano .env
```

### 3. Install Dependencies

```bash
# Create and activate a virtual environment (one-time setup)
python3 -m venv venv
source venv/bin/activate

# Install packages into the venv
pip install -r requirements.txt
```

### 4. Run the Bot

```bash
# Always activate the venv first
source venv/bin/activate
python main.py
```

The bot will:
1. Connect to Manifold and OpenRouter
2. Load your current balance and positions
3. Start making decisions and executing trades automatically
4. Log all activity to both the console and `bot.log`

Press `Ctrl+C` to gracefully stop the bot at any time.

## Project Structure

```
manifold_bot/
├── main.py          # Entry point, main loop, system prompt
├── llm.py           # OpenRouter client, token counting, context compaction
├── tools.py         # All tool implementations and LLM tool definitions
├── manifold.py      # Manifold Markets API client
├── requirements.txt  # Python dependencies
├── .env.example     # Environment variable template
├── .env             # Your API keys (not committed)
├── PLAN.md          # Bot's persistent strategic plan
├── bot.log          # Activity log
└── README.md        # This file
```

## Models

| Priority | Model | Description |
|----------|-------|-------------|
| Primary | `nvidia/nemotron-3-ultra-550b-a55b:free` | Default model, free tier |
| Fallback | `poolside/laguna-m.1:free` | Activated after 3 consecutive Nemotron timeouts |

The bot periodically probes the primary model and switches back when it's available again.

## Tools Reference

| Tool | Description |
|------|-------------|
| `browse_markets(limit, filter)` | Browse recently created markets |
| `search_markets(term, filter, sort)` | Search markets by keyword |
| `get_market(market_id)` | Get detailed market info |
| `place_bet(contract_id, amount, outcome, limit_prob?)` | Place a bet (market or limit order) |
| `sell_position(contract_id, shares?)` | Sell shares in a market |
| `get_positions()` | View balance and current positions |
| `web_search(query, max_results)` | Search the web via DuckDuckGo |
| `web_fetch(url, max_chars)` | Fetch and extract text from a URL |
| `wait(seconds)` | Pause execution (max 90 minutes) |
| `update_plan(new_plan)` | Update the persistent PLAN document |

## Context Compaction

The bot tracks token usage via `tiktoken`. When the conversation exceeds 80% of the 100K token threshold:

1. The most recent 15 messages are kept verbatim
2. Older messages are summarized by the LLM into a compact narrative
3. The system prompt and PLAN document are preserved

This ensures the bot can run indefinitely without losing its accumulated knowledge.

## Logging

All activity is logged to both stdout and `bot.log`:
- Tool calls and their results
- LLM API calls and token usage
- Market research and trade execution
- Errors and retry logic

## Rate Limits

- **Manifold API**: 500 requests/minute per IP
- **OpenRouter Free Tier**: 20 requests/minute, 50 requests/day (or 1,000/day with $10+ credits)
- **DuckDuckGo Search**: No explicit rate limit, but the library includes built-in throttling

## Disclaimer

This bot trades real Mana on Manifold Markets. While Mana is a virtual currency (not real money), the bot will place real bets based on AI decision-making. Use at your own risk. The AI may make poor trading decisions.
