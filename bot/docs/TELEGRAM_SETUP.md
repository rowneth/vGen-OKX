# Telegram + Paper Trading Setup

Run the bot against the real MEXC futures feed with fake money, and get a
Telegram notification on every trigger.

## 1. Create a Telegram bot

1. Open Telegram and chat with **@BotFather**.
2. Send `/newbot`, pick a display name and a username (`..._bot`).
3. Copy the **HTTP API token** it returns (looks like `1234:ABC...`).

## 2. Find your chat id

1. Start a chat with **@userinfobot** and send any message — it replies with your numeric `id`.
2. That numeric id is your `TELEGRAM_CHAT_ID`.
3. Send `/start` to **your new bot** at least once so it is allowed to message you.

## 3. Configure environment variables

Copy `.env.example` → `.env` and fill in:

```env
MEXC_API_KEY=...
MEXC_API_SECRET=...
TELEGRAM_BOT_TOKEN=1234:ABC...
TELEGRAM_CHAT_ID=123456789
```

You can fine-tune which events trigger a message in `config/config.yaml`:

```yaml
notifications:
  telegram:
    send_signals: true      # setup detected + position opened
    send_entries: true      # entry fill
    send_partials: true     # TP1 / TP2 scale-outs
    send_exits: true        # final close (TP / stop / time-stop)
    send_errors: true       # runner errors, data outages
    daily_report_utc_hour: 0
```

## 4. Start the 7-day paper run

```bash
source .venv/bin/activate
cd mexc-bot
python scripts/run_paper.py --duration-days 7
```

On startup you'll receive a `🚀 Bot started` banner. While running you'll get:

- 🎯 **Signal** + ✅ **Entry filled** the instant the strategy opens a trade
- 💰 **TP1 hit** / 💎 **TP2 hit** with partial PnL and the new stop
- 🏁 / 🛑 / ⏱️ **Position closed** on final TP, stop, or time-stop, with net PnL and new equity
- 📊 **Daily report** at the configured UTC hour with trades / WR / gross / fees / net / volume / equity delta
- ❗ **Error** if kline polling or the session throws

The session writes `data/paper_session_state.json` after every poll, so you
can stop (`Ctrl+C`) and resume with `--resume` without losing equity or trade
history.
