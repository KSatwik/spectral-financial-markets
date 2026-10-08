# Spectral Analysis of Financial Markets

A research tool that investigates whether **frequency-domain
(spectral) features** of financial time series provide *incremental* predictive
and economic value beyond conventional time-domain features.

> **My question:** Can rolling spectral features improve market-regime
> detection, volatility/direction forecasting, and systematic investment
> performance after realistic transaction costs?

**Live app:** https://satwik-spectral-financial-markets.streamlit.app/

---


## What This Project Does

Most quantitative finance tools analyze markets in the **time domain** —
looking at returns, volatility, momentum, and moving averages over days and
weeks.

This project instead analyzes markets in the **frequency domain** using the
Fourier transform — asking questions like:

- Are there hidden cycles in the price series?
- Do these cycles change when markets enter different regimes?
- Do cycles help predict tomorrow better than conventional features?

The output is an interactive web app where you can select a market, choose
a rolling window, and immediately see:

- Spectral features over time
- Market regimes overlaid on price
- Walk-forward predictive performance
- Backtest results with realistic transaction costs

**This is a research prototype, not a trading system.** Negative results
are considered valid and informative outcomes.

---

## Research Hypotheses

The project tests five hypotheses. **None are assumed true.**

| # | Hypothesis | Status |
|---|-----------|--------|
| **H1** | Financial time series exhibit changing frequency-domain characteristics across different market regimes. | Under test |
| **H2** | Spectral features contain information not fully captured by conventional time-domain features. | Under test |
| **H3** | Adding spectral features improves out-of-sample prediction of volatility, large moves, or direction. | Under test |
| **H4** | Any predictive improvement translates into improved risk-adjusted investment performance. | Under test |
| **H5** | Any improvement survives across assets, periods, window sizes, and realistic transaction costs. | Under test |

A null or negative result is a **valid** research result and will be reported honestly.

---

## Features

### 📊 Overview Tab
- Price chart (log scale)
- Daily log-return bar chart
- Summary statistics (observations, date range, volatility, cumulative return)

### 📈 Time-Domain Tab
The conventional baseline against which spectral features are compared:
- Rolling momentum (5, 20, 60 days)
- Rolling annualized volatility (5, 20, 60 days)
- Rolling lag-1 autocorrelation
- Rolling skewness and kurtosis

### 🌊 Spectral Tab
The novel contribution:
- Rolling spectral entropy
- Rolling spectral centroid
- Low vs high-frequency energy share
- Rolling dominant period
- **Spectrogram heatmap** showing how frequency content evolves over time

### 🔀 Regimes Tab
Rule-based regime classification using only trailing information:
- **Calm-Bull** (low vol, positive trend)
- **Calm-Bear** (low vol, negative trend)
- **Normal-Bull** / **Normal-Bear**
- **Stressed-Bull** / **Stressed-Bear**
- Mean spectral features shown per regime

### 🎯 Prediction & Backtest Tab
Walk-forward machine learning with strict no-look-ahead rules:
- Three feature sets compared: time-domain only, spectral only, combined
- Model choice: Random Forest or Logistic Regression
- Expanding-window walk-forward validation
- Classification metrics: accuracy, F1, ROC-AUC
- Backtest: long-if-P(up) > 0.5, else cash
- Transaction-cost sensitivity (0–50 bps)
- Feature importance ranking

---

## Methodology

### Data
- **Source:** Yahoo Finance via `yfinance`
- **Adjustment:** Automatically adjusted for splits and dividends
- **Frequency:** Daily
- **Range:** User-selectable (default 2010 to today)

**Supported markets:**

| Region | Ticker | Symbol |
|--------|--------|--------|
| USA | S&P 500 | `SPY` |
| USA | NASDAQ-100 | `QQQ` |
| India | NIFTY 50 | `^NSEI` |
| Europe | STOXX Europe 50 | `^STOXX50E` |
| Japan | Nikkei 225 | `^N225` |
| Commodity | Gold | `GLD` |
| Commodity | Crude Oil | `USO` |
| FX | EUR/USD | `EURUSD=X` |
| Crypto | Bitcoin | `BTC-USD` |

### Returns
Log returns are used throughout:
$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$

### Time-Domain Features (Baseline)

| Category | Features |
|----------|----------|
| Momentum | 5, 10, 20, 60-day returns |
| Volatility | 5, 10, 20, 60-day rolling standard deviation × √252 |
| Statistical | Rolling lag-1 autocorrelation, skewness, kurtosis |
| Drawdown | 60-day cumulative drawdown |
| Trend | Distance from 20-day and 60-day moving averages |

### Spectral Features

For each rolling window of length N (64, 128, 256, or 512), we compute the
real FFT and derive:

| Feature | Definition |
|---------|------------|
| **Dominant frequency** | `argmax(P(f))` — location of peak power |
| **Dominant period** | `1 / dominant frequency` |
| **Spectral entropy** | `-Σ p_i log(p_i)` where `p_i = P_i / ΣP` |
| **Spectral centroid** | `Σ f·P(f) / ΣP(f)` |
| **Low-frequency energy** | Share of power in periods ≥ 20 days |
| **High-frequency energy** | Share of power in periods ≤ 5 days |
| **Spectral concentration** | Share of power in top 10% of bins |

### Validation
- **Expanding-window walk-forward** — no random splits
- **All normalization** done inside training folds only
- **Targets** use `shift(-1)` — strictly future returns
- **Thresholds** for large-move labels use trailing 252-day quantiles

### Models
- **Random Forest** (150 trees, max depth 5, min leaf 20)
- **Logistic Regression** (L2, C=1.0)

### Backtest Assumptions
- Position: long if `P(up) > 0.5`, else cash
- Execution lag: one bar (no same-bar close-to-close execution)
- Transaction costs: 0, 5, 10, 25, or 50 bps on position changes
- No shorting, no leverage, no slippage model

---

