# MarketAI Repository Model Comparison

## Scientific grouping

| Family | Repositories | Primary method | MarketAI role |
|---|---|---|---|
| Financial sentiment | FinBERT | BERT encoder further trained/fine-tuned for 3-class financial sentiment | Baseline NLP classifier |
| Financial LLM | FinGPT, FinLLMs, PIXIU, FLANG, InvestLM | Decoder/LLM fine-tuning, instruction tuning, financial task evaluation | Evidence extraction, classification, summarization, research |
| Probabilistic TS foundation | Chronos, TimesFM, Uni2TS, FinCast, Timer-XL | Pretrained sequence/time-series models | Controlled forecasting tournament |
| Financial K-line foundation | Kronos, Kronos1, Conductor | Financial candlestick tokenization + autoregressive Transformer lineage | Forecasting tournament; Conductor is engineering distribution |
| Stock prediction | StockMixer, StockCL | Stock-specific deep/representation learning | Controlled benchmark |
| General forecasting research | Financial-Time-Series | Financial forecasting experiments and reference methods | Benchmark/source of candidate baselines |
| Reinforcement learning | FinRL | Trading environments + RL policies | Research-only until simulator/reward validity is established |
| Portfolio/risk | Skfolio | Portfolio optimization, risk modeling and allocation | Post-prediction risk/portfolio layer |
| Backtesting | VectorBT, Backtrader | Vectorized and event-driven historical simulation | Research engines; one authoritative production simulator only |
| Execution | NautilusTrader, Lumibot, Hummingbot, Freqtrade | Event-driven/order/exchange abstractions | Narrowly selected adapters; Nautilus candidate primary simulator/execution boundary |
| Agents | OpenAlice, MIT-TradingAgents, Vibe-Trading, Hermes-Agent, Ornith-1 | Tool-using multi-agent workflows | Bounded research/evidence orchestration |
| Local inference | Kimi-K3-in-C, Turbo-Fieldfare | Runtime/memory/performance engineering | Optional inference infrastructure |
| Security | Shannon, Numbat | SAST, testing, security workflows | Security gate/tooling, isolated from prediction path |

## Baseline rule

Every numerical forecast model must compete on an identical point-in-time dataset against naive/random-walk/seasonal-naive/moving-average/momentum plus linear/logistic and tree-based baselines where the target permits them.

The tournament must record metrics by asset, horizon, regime and evaluation window. A model is not promoted because it is larger, newer or a foundation model.

## Interface contract

All forecast candidates must implement the same conceptual contract:

```text
fit(dataset_snapshot)
predict(feature_snapshot, horizon)
metadata()
```

`metadata()` must expose model version, code commit, training data snapshot, feature version, training cutoff and inference environment.

## Important non-equivalence findings

- FinBERT produces text sentiment; it is not a price predictor.
- Kronos/Chronos/TimesFM/etc. are forecasting candidates; their public benchmark behavior cannot be assumed to transfer to NIFTY-50.
- NautilusTrader/Lumibot/Backtrader/VectorBT are infrastructure/research engines, not competing alpha models.
- OpenAlice/Hermes/MIT-TradingAgents are workflow/agent systems and require governance around tools and evidence.
- Kimi-K3-in-C is an optimized inference implementation; it does not add financial predictive validity by itself.
