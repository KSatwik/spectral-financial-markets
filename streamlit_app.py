"""
Spectral Analysis of Financial Markets — Streamlit research app.

Frequency-domain features for regime detection, predictive experiments, and systematic backtesting with realistic transaction costs.
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from numpy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Spectral Financial Markets",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TICKERS = {
    "S&P 500 (SPY)": "SPY",
    "NASDAQ-100 (QQQ)": "QQQ",
    "NIFTY 50 (^NSEI)": "^NSEI",
    "STOXX Europe 50 (^STOXX50E)": "^STOXX50E",
    "Nikkei 225 (^N225)": "^N225",
    "Gold (GLD)": "GLD",
    "Crude Oil (USO)": "USO",
    "EUR/USD (EURUSD=X)": "EURUSD=X",
    "Bitcoin (BTC-USD)": "BTC-USD",
}

TD_FEATURES = [
    "mom_5", "mom_10", "mom_20", "mom_60",
    "vol_5", "vol_10", "vol_20", "vol_60",
    "autocorr_20", "skew_20", "kurt_20",
    "dd_60", "dist_ma_20", "dist_ma_60",
]
SP_FEATURES = [
    "dom_freq", "dom_period", "spec_entropy", "spec_centroid",
    "low_energy", "high_energy", "spec_concentration",
]

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, start: str, end: str):
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    return df[["Close"]].dropna()


# --------------------------------------------------------------------------
# Time-domain features
# --------------------------------------------------------------------------
def time_domain_features(ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=ret.index)
    for w in (5, 10, 20, 60):
        df[f"mom_{w}"] = ret.rolling(w).sum()
    for w in (5, 10, 20, 60):
        df[f"vol_{w}"] = ret.rolling(w).std() * np.sqrt(252)
    df["autocorr_20"] = ret.rolling(20).corr(ret.shift(1))
    df["skew_20"] = ret.rolling(20).skew()
    df["kurt_20"] = ret.rolling(20).kurt()
    cum = ret.cumsum()
    df["dd_60"] = cum - cum.rolling(60, min_periods=1).max()
    price = np.exp(cum)
    for w in (20, 60):
        ma = price.rolling(w).mean()
        df[f"dist_ma_{w}"] = price / ma - 1.0
    return df


# --------------------------------------------------------------------------
# Spectral features
# --------------------------------------------------------------------------
def spectral_features_window(x: np.ndarray) -> dict:
    nan_out = dict(
        dom_freq=np.nan, dom_period=np.nan, spec_entropy=np.nan,
        spec_centroid=np.nan, low_energy=np.nan, high_energy=np.nan,
        spec_concentration=np.nan,
    )
    x = x - x.mean()
    N = len(x)
    if N < 8 or np.allclose(x, 0):
        return nan_out
    P = np.abs(rfft(x)) ** 2
    f = rfftfreq(N, d=1.0)
    P, f = P[1:], f[1:]              # drop DC
    total = P.sum()
    if total <= 0:
        return nan_out
    p = P / total
    dom_idx = int(np.argmax(P))
    dom_freq = float(f[dom_idx])
    dom_period = float(1.0 / dom_freq) if dom_freq > 0 else np.inf
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    centroid = float(np.sum(f * p))
    low_energy = float(P[f <= 0.05].sum() / total)   # period >= 20 days
    high_energy = float(P[f >= 0.20].sum() / total)  # period <= 5 days
    k = max(1, len(P) // 10)
    conc = float(np.sort(P)[-k:].sum() / total)
    return dict(
        dom_freq=dom_freq, dom_period=dom_period, spec_entropy=entropy,
        spec_centroid=centroid, low_energy=low_energy,
        high_energy=high_energy, spec_concentration=conc,
    )


def rolling_spectral(ret: pd.Series, window: int) -> pd.DataFrame:
    vals = ret.values
    idx = ret.index
    rows, dates = [], []
    for i in range(window, len(vals) + 1):
        rows.append(spectral_features_window(vals[i - window:i]))
        dates.append(idx[i - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


# --------------------------------------------------------------------------
# Build dataset
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching data and computing features...")
def build_dataset(ticker: str, start: str, end: str, spec_window: int):
    prices = load_prices(ticker, start, end)
    if prices is None or len(prices) < 300:
        return None
    close = prices["Close"].astype(float)
    ret = np.log(close / close.shift(1)).dropna()

    td = time_domain_features(ret)
    sp = rolling_spectral(ret, spec_window)

    df = td.join(sp, how="inner")
    df["ret"] = ret.reindex(df.index)
    df["close"] = close.reindex(df.index)

    # ---- Targets (strictly forward, no look-ahead) ----
    future_ret = df["ret"].shift(-1)

    # Direction of next-day return
    df["target_up"] = (
        (future_ret > 0).astype(float).where(future_ret.notna())
    )

    # Large-move flag using *trailing* 252-day 75th percentile
    thresh = df["ret"].abs().rolling(252, min_periods=60).quantile(0.75)
    df["target_large"] = (
        (future_ret.abs() > thresh).astype(float)
        .where(future_ret.notna() & thresh.notna())
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# --------------------------------------------------------------------------
# Regimes (rule-based, no look-ahead)
# --------------------------------------------------------------------------
def assign_regimes(df: pd.DataFrame) -> pd.Series:
    vol = df["vol_20"]
    mom = df["mom_60"]
    reg = pd.Series(index=df.index, dtype=object)
    reg[(vol < 0.15) & (mom >= 0)] = "Calm-Bull"
    reg[(vol < 0.15) & (mom < 0)] = "Calm-Bear"
    reg[(vol >= 0.15) & (vol < 0.25) & (mom >= 0)] = "Normal-Bull"
    reg[(vol >= 0.15) & (vol < 0.25) & (mom < 0)] = "Normal-Bear"
    reg[(vol >= 0.25) & (mom >= 0)] = "Stressed-Bull"
    reg[(vol >= 0.25) & (mom < 0)] = "Stressed-Bear"
    return reg


# --------------------------------------------------------------------------
# Walk-forward validation
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Running walk-forward validation...")
def walk_forward(df: pd.DataFrame, feature_cols: tuple, target_col: str,
                 model_name: str, initial_train: int, step: int):
    X = df[list(feature_cols)]
    y = df[target_col]
    n = len(X)
    preds = np.full(n, np.nan)
    importances = []

    for start in range(initial_train, n, step):
        end = min(start + step, n)
        X_tr, y_tr = X.iloc[:start], y.iloc[:start]
        X_te = X.iloc[start:end]
        if len(X_tr) < 60 or y_tr.nunique() < 2:
            continue
        scaler = StandardScaler().fit(X_tr)
        if model_name == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=150, max_depth=5, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            )
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0)
        try:
            clf.fit(scaler.transform(X_tr), y_tr)
        except Exception:
            continue
        preds[start:end] = clf.predict_proba(scaler.transform(X_te))[:, 1]
        if hasattr(clf, "feature_importances_"):
            importances.append(clf.feature_importances_)

    preds_s = pd.Series(preds, index=X.index)
    imp = (
        pd.Series(np.mean(importances, axis=0), index=list(feature_cols))
        if importances else None
    )
    return preds_s, imp


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def run_backtest(signal: pd.Series, ret: pd.Series, cost_bps: float):
    pos = (signal > 0.5).astype(float)
    pos_exec = pos.shift(1).fillna(0.0)             # held during return period
    turnover = pos_exec.diff().abs().fillna(0.0)
    cost = turnover * (cost_bps / 10000.0)
    strat_ret = pos_exec * ret - cost
    return strat_ret, pos_exec, turnover


def perf_metrics(returns: pd.Series, periods: int = 252) -> dict:
    r = returns.dropna()
    if len(r) < 5:
        return {}
    cum = (1 + r).cumprod()
    ann_ret = cum.iloc[-1] ** (periods / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    # Proper downside deviation (target = 0)
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean()) * np.sqrt(periods)
    sortino = ann_ret / downside if downside > 0 else np.nan
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    return {
        "Cum. Return": cum.iloc[-1] - 1,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "Hit Rate": (r > 0).mean(),
    }


def classification_metrics(preds: pd.Series, y: pd.Series) -> dict:
    m = preds.notna() & y.notna()
    if m.sum() < 20:
        return {}
    p, t = preds[m], y[m]
    pred_lbl = (p > 0.5).astype(int)
    out = {
        "N": int(m.sum()),
        "Accuracy": accuracy_score(t, pred_lbl),
        "F1": f1_score(t, pred_lbl, zero_division=0),
    }
    try:
        out["ROC-AUC"] = roc_auc_score(t, p)
    except ValueError:
        out["ROC-AUC"] = np.nan
    return out


# --------------------------------------------------------------------------
# Spectrogram
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def spectrogram(ret_values, ret_index, window: int, stride: int = 5):
    vals = ret_values
    idx = ret_index
    mats, dates, freqs = [], [], None
    for i in range(window, len(vals) + 1, stride):
        x = vals[i - window:i]
        x = x - x.mean()
        P = np.abs(rfft(x)) ** 2
        f = rfftfreq(window, d=1.0)
        P, f = P[1:], f[1:]
        if freqs is None:
            freqs = f
        mats.append(P)
        dates.append(idx[i - 1])
    if not mats:
        return None, None, None
    return np.array(freqs), pd.DatetimeIndex(dates), np.array(mats).T


# ==========================================================================
# UI
# ==========================================================================
st.title("📈 Spectral Analysis of Financial Markets")
st.caption(
    "Frequency-domain features for regime detection, prediction, and "
    "systematic investment — evaluated under strict walk-forward validation."
)

with st.sidebar:
    st.header("Configuration")
    ticker_label = st.selectbox("Market", list(TICKERS.keys()))
    ticker = TICKERS[ticker_label]
    start = st.date_input("Start date", pd.Timestamp("2010-01-01"))
    end = st.date_input("End date", pd.Timestamp.today())
    spec_window = st.select_slider(
        "Spectral window (days)", options=[64, 128, 256, 512], value=128
    )
    model_name = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    task_label = st.selectbox(
        "Prediction task",
        ["Direction (next-day sign)", "Large move (|ret| > trailing 75th pct)"],
    )
    cost_bps = st.slider("Transaction cost (bps)", 0, 50, 10, step=5)
    st.divider()
    st.caption(
        "⚠️ Research prototype. No look-ahead: features and models use only "
        "information available before the prediction date."
    )

data = build_dataset(ticker, str(start), str(end), spec_window)
if data is None or len(data) < 300:
    st.error(
        "Not enough data. Try a longer date range, a different ticker, or a "
        "smaller spectral window."
    )
    st.stop()

is_direction_task = task_label.startswith("Direction")
target_col = "target_up" if is_direction_task else "target_large"

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_td, tab_sp, tab_reg, tab_pred = st.tabs(
    ["Overview", "Time-Domain", "Spectral", "Regimes", "Prediction & Backtest"]
)

# --- Overview -------------------------------------------------------------
with tab_overview:
    st.subheader(f"{ticker_label} — overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(data):,}")
    c2.metric("Range", f"{data.index[0].date()} → {data.index[-1].date()}")
    c3.metric("Ann. volatility (20d, last)",
              f"{data['vol_20'].iloc[-1]:.1%}")
    c4.metric("Cum. return",
              f"{(np.exp(data['ret'].sum()) - 1):.1%}")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Price (log scale)", "Daily log returns"),
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["close"], name="Close",
                   line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig.update_yaxes(type="log", row=1, col=1)
    fig.add_trace(
        go.Bar(x=data.index, y=data["ret"], name="Log return",
               marker_color=np.where(data["ret"] >= 0, "#2ca02c", "#d62728")),
        row=2, col=1,
    )
    fig.update_layout(height=560, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Data notes.** Adjusted close prices via `yfinance` (auto-adjusted for
        splits and dividends). Log returns are used throughout. Data quality
        limitations (survivorship, corporate actions, timezone alignment) are
        inherited from the source.
        """
    )

# --- Time-domain ----------------------------------------------------------
with tab_td:
    st.subheader("Time-domain baseline features")
    st.caption(
        "These constitute the conventional baseline against which spectral "
        "features are compared. Nothing here should be considered novel."
    )
    sub1, sub2 = st.columns(2)
    with sub1:
        fig = go.Figure()
        for w in (5, 20, 60):
            fig.add_trace(go.Scatter(x=data.index, y=data[f"mom_{w}"],
                                     name=f"Momentum {w}d"))
        fig.update_layout(title="Rolling momentum", height=340,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with sub2:
        fig = go.Figure()
        for w in (5, 20, 60):
            fig.add_trace(go.Scatter(x=data.index, y=data[f"vol_{w}"],
                                     name=f"Vol {w}d"))
        fig.update_layout(title="Rolling annualised volatility", height=340,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    sub3, sub4 = st.columns(2)
    with sub3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["autocorr_20"],
                                 name="Autocorr(1) 20d"))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(title="Rolling lag-1 autocorrelation",
                          height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with sub4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["skew_20"],
                                 name="Skew 20d"))
        fig.add_trace(go.Scatter(x=data.index, y=data["kurt_20"],
                                 name="Excess kurt 20d"))
        fig.update_layout(title="Rolling skewness & kurtosis",
                          height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --- Spectral -------------------------------------------------------------
with tab_sp:
    st.subheader("Frequency-domain features")
    st.markdown(
        f"""
        For every rolling window of **{spec_window}** daily log returns we
        compute the real FFT and derive:

        - **Dominant frequency / period** — location of peak power
        - **Spectral entropy** — dispersion of power across frequencies
        - **Spectral centroid** — power-weighted mean frequency
        - **Low-energy ratio** — power in periods ≥ 20 days
        - **High-energy ratio** — power in periods ≤ 5 days
        - **Spectral concentration** — share of power in top-10% of bins
        """
    )

    sub1, sub2 = st.columns(2)
    with sub1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["spec_entropy"],
                                 name="Spectral entropy",
                                 line=dict(color="#9467bd")))
        fig.update_layout(title="Rolling spectral entropy",
                          height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with sub2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["spec_centroid"],
                                 name="Spectral centroid",
                                 line=dict(color="#ff7f0e")))
        fig.update_layout(title="Rolling spectral centroid (cycles/day)",
                          height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    sub3, sub4 = st.columns(2)
    with sub3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["low_energy"],
                                 name="Low-frequency energy"))
        fig.add_trace(go.Scatter(x=data.index, y=data["high_energy"],
                                 name="High-frequency energy"))
        fig.update_layout(title="Low vs high-frequency energy share",
                          height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with sub4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data["dom_period"].clip(upper=spec_window),
                                 name="Dominant period (days)"))
        fig.update_layout(title="Rolling dominant period",
                          height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Rolling spectrogram (log power)")
    f, d, M = spectrogram(
        data["ret"].values, data.index.values, spec_window, stride=5
    )
    if M is not None:
        fig = go.Figure(go.Heatmap(
            x=d, y=f, z=np.log10(M + 1e-12),
            colorscale="Viridis",
            colorbar=dict(title="log10 power"),
        ))
        fig.update_yaxes(title="Frequency (cycles/day)")
        fig.update_layout(height=420, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The heatmap shows how the spectral profile evolves over time. "
            "Regime shifts typically appear as horizontal band changes."
        )
    else:
        st.info("Not enough data for a spectrogram.")

# --- Regimes --------------------------------------------------------------
with tab_reg:
    st.subheader("Market regimes")
    st.caption(
        "Regimes are rule-based, using only trailing information: 20-day "
        "annualised volatility bands (< 15%, 15–25%, ≥ 25%) crossed with the "
        "sign of the 60-day cumulative return."
    )
    reg = assign_regimes(data)
    data_reg = data.assign(regime=reg).dropna(subset=["regime"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Price with regime overlay",
                        "Realised volatility (20d)"),
    )
    fig.add_trace(
        go.Scatter(x=data_reg.index, y=data_reg["close"], name="Close",
                   line=dict(color="#333")),
        row=1, col=1,
    )
    color_map = {
        "Calm-Bull": "#2ca02c", "Calm-Bear": "#98df8a",
        "Normal-Bull": "#1f77b4", "Normal-Bear": "#aec7e8",
        "Stressed-Bull": "#ff7f0e", "Stressed-Bear": "#d62728",
    }
    for name in data_reg["regime"].unique():
        sub = data_reg.loc[data_reg["regime"] == name]
        fig.add_trace(
            go.Scatter(x=sub.index, y=sub["close"],
                       mode="markers", name=name,
                       marker=dict(size=4, color=color_map.get(name, "grey"))),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=data_reg.index, y=data_reg["vol_20"], name="Vol 20d",
                   line=dict(color="#7f7f7f")),
        row=2, col=1,
    )
    fig.update_layout(height=600, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Mean spectral features by regime")
    agg = data_reg.groupby("regime")[SP_FEATURES].mean().round(4)
    st.dataframe(agg, use_container_width=True)
    st.caption(
        "If spectral characteristics were constant across regimes, these "
        "rows would be nearly identical. Differences here are the first "
        "evidence relevant to H1 and H2."
    )

# --- Prediction & Backtest ------------------------------------------------
with tab_pred:
    st.subheader("Walk-forward prediction and backtesting")
    n = len(data)
    # Robust fold setup for both short and long histories
    initial_train = min(max(200, int(0.4 * n)), n - 100)
    step = max(20, min(126, (n - initial_train) // 3))

    st.markdown(
        f"""
        **Setup.** Expanding-window walk-forward: first {initial_train} days
        are used for initial training, then the model is retrained every
        {step} days. All features and normalisation statistics come strictly
        from the past. Target: **{target_col}**.
        """
    )
    if not is_direction_task:
        st.info(
            "The **Large-move** task is classification-only. A directional "
            "backtest from a large-move probability would be meaningless "
            "without a separate direction signal."
        )

    feature_sets = {
        "Time-domain only": tuple(TD_FEATURES),
        "Spectral only": tuple(SP_FEATURES),
        "Combined": tuple(TD_FEATURES + SP_FEATURES),
    }

    if st.button("▶ Run walk-forward experiment", type="primary"):
        st.session_state["run_pred"] = True

    if st.session_state.get("run_pred", False):
        results, preds_dict, imp_dict = {}, {}, {}
        for label, cols in feature_sets.items():
            preds, imp = walk_forward(
                data, cols, target_col, model_name, initial_train, step
            )
            preds_dict[label] = preds
            imp_dict[label] = imp
            results[label] = classification_metrics(preds, data[target_col])

        st.markdown("### Out-of-sample classification performance")
        cls_df = pd.DataFrame(results).T
        if not cls_df.empty and not cls_df.isna().all().all():
            st.dataframe(
                cls_df.style.format({
                    "N": "{:,.0f}", "Accuracy": "{:.3f}",
                    "F1": "{:.3f}", "ROC-AUC": "{:.3f}",
                }, na_rep="–"),
                use_container_width=True,
            )
        else:
            st.warning(
                "No valid out-of-sample predictions. Try a longer date range "
                "or a different asset."
            )

        st.caption(
            "The scientific question is whether **Combined** improves over "
            "**Time-domain only**. If it does not, spectral features add no "
            "incremental predictive value (H3 not supported)."
        )

        # ---------------- Backtest (direction task only) ----------------
        if is_direction_task and not cls_df.empty:
            st.markdown("### Backtest (long-if-P(up) > 0.5, else cash)")
            bh_ret = data["ret"]
            curves = {"Buy & Hold": (1 + bh_ret).cumprod()}
            metrics_rows = {"Buy & Hold": perf_metrics(bh_ret)}

            for label in feature_sets:
                strat_ret, _, to = run_backtest(
                    preds_dict[label], data["ret"], cost_bps
                )
                curves[label] = (1 + strat_ret).cumprod()
                m = perf_metrics(strat_ret)
                m["Turnover (ann.)"] = to.mean() * 252
                metrics_rows[label] = m

            fig = go.Figure()
            for name, c in curves.items():
                fig.add_trace(go.Scatter(x=c.index, y=c.values, name=name))
            fig.update_layout(
                title=f"Cumulative returns — {cost_bps} bps transaction cost",
                height=460, yaxis_title="Growth of $1",
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            metric_df = pd.DataFrame(metrics_rows).T
            fmt = {
                "Cum. Return": "{:.1%}", "Ann. Return": "{:.1%}",
                "Ann. Vol": "{:.1%}", "Max Drawdown": "{:.1%}",
                "Hit Rate": "{:.1%}", "Turnover (ann.)": "{:.1f}",
                "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}",
            }
            fmt = {k: v for k, v in fmt.items() if k in metric_df.columns}
            st.dataframe(
                metric_df.style.format(fmt, na_rep="–"),
                use_container_width=True,
            )

            st.markdown("### Transaction-cost sensitivity")
            costs = [0, 5, 10, 25, 50]
            cost_table = {}
            for label in feature_sets:
                row = {}
                for cb in costs:
                    sr, _, _ = run_backtest(preds_dict[label], data["ret"], cb)
                    row[f"{cb} bps"] = perf_metrics(sr).get("Sharpe", np.nan)
                cost_table[label] = row
            cost_df = pd.DataFrame(cost_table).T
            st.dataframe(
                cost_df.style.format("{:.2f}", na_rep="–"),
                use_container_width=True,
            )
            st.caption("Sharpe ratio as a function of assumed round-trip cost.")

        # ---------------- Feature importance (if RF) ----------------
        if imp_dict.get("Combined") is not None:
            st.markdown("### Feature importance (combined model, RF)")
            imp = imp_dict["Combined"].sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker_color=[
                    "#d62728" if c in SP_FEATURES else "#1f77b4"
                    for c in imp.index
                ],
            ))
            fig.update_layout(
                height=440, margin=dict(t=20, b=10),
                xaxis_title="Mean importance across folds",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red bars are spectral features; blue are time-domain.")

        # ---------------- Honest summary ----------------
        st.markdown("---")
        st.markdown("### Interpretation")
        td_auc = results.get("Time-domain only", {}).get("ROC-AUC", np.nan)
        comb_auc = results.get("Combined", {}).get("ROC-AUC", np.nan)
        sp_auc = results.get("Spectral only", {}).get("ROC-AUC", np.nan)
        if np.isnan(td_auc) or np.isnan(comb_auc):
            st.info("Not enough folds to draw conclusions.")
        else:
            delta = comb_auc - td_auc
            if delta > 0.01:
                verdict = (
                    "Combined features **outperform** the time-domain "
                    "baseline by more than 1 point of ROC-AUC. This is "
                    "consistent with H3 — but should be confirmed across "
                    "markets, windows, and cost assumptions."
                )
            elif delta < -0.01:
                verdict = (
                    "Adding spectral features **degrades** performance. "
                    "Consistent with H3 being rejected on this slice."
                )
            else:
                verdict = (
                    "Differences are within noise (< 1 pt ROC-AUC). Under "
                    "this configuration, spectral features do **not** add "
                    "incremental predictive value."
                )
            st.markdown(
                f"""
                - Time-domain-only ROC-AUC: **{td_auc:.3f}**
                - Spectral-only ROC-AUC: **{sp_auc:.3f}**
                - Combined ROC-AUC: **{comb_auc:.3f}**

                {verdict}

                Statistical significance ≠ predictive significance ≠
                economic significance. A positive backtest is not evidence
                that a strategy will work live.
                """
            )
    else:
        st.info(
            "Click **Run walk-forward experiment** to train models and "
            "evaluate them out-of-sample. This may take ~10–30 seconds."
        )

st.markdown("---")
st.caption(
    "Built as a research prototype. All results are conditional on the "
    "chosen asset, period, window, model, and cost assumptions. "
    "Negative results are valid and informative."
)
