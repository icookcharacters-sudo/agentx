# real ai — Autonomous Multi-Agent Studio

A premium Streamlit app for a fully autonomous **3-model loop** (Planner, Researcher, Executor) using the **Kimi API** (`kimi-k2.6`) via an OpenAI-compatible endpoint.

## Features

- One task input → autonomous multi-agent execution
- Live terminal sandbox with timestamps and role pills
- Structured JSON outputs + raw JSON viewer
- Validation + fallback for executor actions
- Persistent workspace and memory (`.real_ai/`)
- Stagnation handling + completion control
- Single iteration or run-until-complete modes
- **Evolution Mode** for recursive self-improvement with diff preview and one-click apply
- Premium UI (deep black + cyan gradients, micro-animations, metric cards)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kimi configuration

- API key: `KIMI_API_KEY` or paste directly in the sidebar
- Base URL: `https://api.moonshot.ai/v1`
- Model: `kimi-k2.6`

> If no API key is provided, the app runs in local simulation mode so you can test UX and loop logic.
