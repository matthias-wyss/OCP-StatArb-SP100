# trading.py
"""
Intraday lead-lag trading strategy based on the Optimal Causal Path (OCP) algorithm.

This module implements a trading strategy that follows the logic of the original OCP paper:
  - One-day formation / one-day trading structure
  - OCP used to select top stable leader–follower pairs with non-zero lag
  - Trading signals generated from Bollinger bands on leader returns
  - Positions opened in the follower stock against a market index (market-neutral)
  - Exits governed by an expected reaction window derived from OCP lag and its fluctuation

The implementation builds on the structure used in your team's notebook:
  - Polars + NumPy
  - Dataclass for parameters
  - NaN-safe rolling statistics
  - Per-day alignment of minute returns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import date, datetime


import numpy as np
import polars as pl


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class OCPTradeParams:
    """
    Hyperparameters for the OCP-based intraday trading strategy.

    Attributes
    ----------
    lookback : int
        Rolling window (in minutes) for computing Bollinger bands on leader returns.
        The original paper uses d = 20.
    k : float
        Width of the Bollinger bands (in standard deviations). The paper uses k = 2.5.
    tc_bps : float
        Transaction cost per round-trip trade, in basis points, at the *portfolio*
        level. The paper uses about 4 bps per share per round-trip; here we apply
        tc_bps as a single scalar cost per closed trade.
    r_bps : float
        Economic return threshold in basis points. A trade is only opened when the
        leader’s absolute minute return exceeds this threshold, and closed once the
        follower-vs-index return exceeds it within the expected lag window.
    max_lag_minutes : int
        Maximum OCP lag (in minutes) that is considered tradable. Larger lags are
        treated as unreliable and the corresponding pairs are ignored.
    ci_z : float
        Z-score for the confidence interval around the OCP lag. The paper uses a
        99.5% confidence interval; z ≈ 2.8 is a reasonable approximation.
    enter_on_next_bar : bool
        If True, apply one-bar execution delay after a signal, to avoid same-bar
        look-ahead bias.
    """

    lookback: int = 20
    k: float = 2.5
    tc_bps: float = 4.0
    r_bps: float = 4.0
    max_lag_minutes: int = 30
    ci_z: float = 2.8
    enter_on_next_bar: bool = True


# =============================================================================
# Utilities
# =============================================================================

def rolling_mean_std_nan_safe(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling mean and std over a fixed window, NaN-safe.

    For each index t, the window is x[t-window+1 : t+1] (inclusive), provided there
    are at least `window` valid (finite) points; otherwise, the result is NaN.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array (e.g. returns).
    window : int
        Rolling window length in elements.

    Returns
    -------
    mu : np.ndarray
        Rolling mean, same shape as x.
    sigma : np.ndarray
        Rolling standard deviation, same shape as x.
    """
    n = len(x)
    mu = np.full(n, np.nan, dtype=float)
    sigma = np.full(n, np.nan, dtype=float)
    if window <= 1 or n == 0:
        return mu, sigma

    # Cumulative sums ignoring NaNs
    valid = np.isfinite(x).astype(float)
    x_clean = np.where(np.isfinite(x), x, 0.0)

    csum = np.cumsum(x_clean)
    csum2 = np.cumsum(x_clean * x_clean)
    ccount = np.cumsum(valid)

    for t in range(window - 1, n):
        start = t - window + 1
        cnt = ccount[t] - (ccount[start - 1] if start > 0 else 0.0)
        if cnt < window:  # require full window of valid data
            continue
        s = csum[t] - (csum[start - 1] if start > 0 else 0.0)
        s2 = csum2[t] - (csum2[start - 1] if start > 0 else 0.0)
        m = s / cnt
        v = s2 / cnt - m * m
        v = max(v, 0.0)
        mu[t] = m
        sigma[t] = np.sqrt(v)

    return mu, sigma


def load_aligned_returns_for_day(
    returns_dir: Path,
    tickers: Iterable[str],
    trade_day,
    returns_col: str = "mid_price_return",
    timestamp_col: str = "timestamp",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and align minute returns for a set of tickers on a given trading day.

    This version:
      - avoids Python datetime/date completely,
      - lets Polars handle the date comparison,
      - avoids DuplicateError by dropping 'timestamp_right'.
    """
    # Represent the target date as a Polars literal cast to Date
    trade_day_lit = pl.lit(trade_day).cast(pl.Date)

    tickers = list(dict.fromkeys(tickers))  # unique, stable order
    df_all: Optional[pl.DataFrame] = None

    for ticker in tickers:
        path = returns_dir / f"{ticker}.parquet"
        if not path.exists():
            continue

        df = (
            pl.read_parquet(path)
            # Cast timestamp to Date and compare to trade_day_lit
            .with_columns(pl.col(timestamp_col).dt.date().alias("_day"))
            .filter(pl.col("_day") == trade_day_lit)
            .select([timestamp_col, returns_col])
            .rename({returns_col: ticker})
        )

        if df.height == 0:
            continue

        if df_all is None:
            df_all = df
        else:
            ts_right = f"{timestamp_col}_right"
            if ts_right in df_all.columns:
                df_all = df_all.drop(ts_right)

            df_all = df_all.join(df, on=timestamp_col, how="outer")

    if df_all is None or df_all.height == 0:
        raise ValueError(f"No data found for {trade_day} in {returns_dir}")

    # Clean up any leftover timestamp_right and helper columns
    ts_right = f"{timestamp_col}_right"
    drop_cols = []
    if ts_right in df_all.columns:
        drop_cols.append(ts_right)
    if "_day" in df_all.columns:
        drop_cols.append("_day")
    if drop_cols:
        df_all = df_all.drop(drop_cols)

    df_all = df_all.sort(timestamp_col)
    timestamps = df_all[timestamp_col].to_numpy()
    df_all = df_all.fill_null(0.0)

    aligned: Dict[str, np.ndarray] = {}
    for ticker in tickers:
        if ticker in df_all.columns:
            aligned[ticker] = df_all[ticker].to_numpy()

    return timestamps, aligned




# =============================================================================
# Core trading logic (per pair, per day)
# =============================================================================

def trade_one_pair_one_day_ocp_style(
    leader_ret: np.ndarray,
    follower_ret: np.ndarray,
    index_ret: Optional[np.ndarray],
    lag_hat: float,
    sigma_l: float,
    params: OCPTradeParams,
) -> Tuple[float, int, int]:
    """
    Trade a single leader–follower pair for one trading day following the OCP paper logic.

    The algorithm:
      - Computes Bollinger bands on leader returns (rolling lookback, width k).
      - Uses an economic threshold r (in returns) tied to transaction costs.
      - When leader returns break the bands and exceed r, opens a position:
          * Long follower / short index on positive leader shock
          * Short follower / long index on negative leader shock
      - Exits when the follower-vs-index cumulative return meets or exceeds r
        within an expected reaction window derived from (lag_hat, sigma_l),
        or at the end of the day.
      - PnL is computed on follower - index leg, and a fixed cost is charged
        per completed round trip.

    Parameters
    ----------
    leader_ret : np.ndarray
        Minute-by-minute returns of the leader on the trading day.
    follower_ret : np.ndarray
        Minute-by-minute returns of the follower on the trading day.
    index_ret : Optional[np.ndarray]
        Minute-by-minute returns of the market index. If None, the follower is
        traded unhedged (not recommended, but allowed).
    lag_hat : float
        Estimated OCP lag (in minutes) from the formation day. Only positive lags
        are tradable (leader leads follower).
    sigma_l : float
        OCP lag fluctuation (standard deviation of lag along the optimal path).
    params : OCPTradeParams
        Trading parameters.

    Returns
    -------
    pnl : float
        Net PnL (in “return units” for one unit notional).
    n_entries : int
        Number of entries (opening trades).
    n_exits : int
        Number of exits (closing trades).
    """
    # Basic sanity checks
    n = min(len(leader_ret), len(follower_ret))
    if n == 0:
        return 0.0, 0, 0

    # We require a positive, reasonably bounded lag
    if lag_hat <= 0 or lag_hat > params.max_lag_minutes:
        return 0.0, 0, 0

    # Clean arrays: replace NaN with 0 for returns
    rL = np.where(np.isfinite(leader_ret[:n]), leader_ret[:n], 0.0)
    rF = np.where(np.isfinite(follower_ret[:n]), follower_ret[:n], 0.0)

    if index_ret is not None and len(index_ret) >= n:
        rI = np.where(np.isfinite(index_ret[:n]), index_ret[:n], 0.0)
    else:
        # If no index provided, treat index return as 0 (unhedged follower)
        rI = np.zeros_like(rF)

    # Precompute follower-minus-index returns and cumulative sums for fast PnL
    rFI = rF - rI
    cum_rFI = np.cumsum(rFI)

    # Rolling stats on leader
    muL, sigL = rolling_mean_std_nan_safe(rL, params.lookback)

    # Economic threshold and transaction cost in absolute return units
    r_thresh = params.r_bps / 10000.0
    tc = params.tc_bps / 10000.0

    # Precompute integer lag window (min and max offset after entry)
    # CI: [lag_hat - z*sigma_l, lag_hat + z*sigma_l]
    if sigma_l is None or not np.isfinite(sigma_l):
        sigma_l = 0.0

    off_min = int(np.floor(lag_hat - params.ci_z * sigma_l))
    off_max = int(np.ceil(lag_hat + params.ci_z * sigma_l))
    off_min = max(1, off_min)  # at least 1 minute after entry
    off_max = max(off_min, off_max)
    # Clip by a hard maximum to avoid silly windows
    off_max = min(off_max, params.max_lag_minutes)

    # State variables
    pos = 0  # +1 long follower/short index, -1 short follower/long index
    entry_idx: Optional[int] = None
    earliest_exit_idx: Optional[int] = None
    deadline_idx: Optional[int] = None
    pending_pos: Optional[int] = None  # for enter_on_next_bar logic

    pnl = 0.0
    n_entries = 0
    n_exits = 0

    # Main intraday loop
    for t in range(n):
        # Apply pending position change at the open of bar t (one-bar delay)
        if pending_pos is not None:
            # Charge transaction cost for a completed roundtrip only
            if pos == 0 and pending_pos != 0:
                # Opening a new trade
                pos = pending_pos
                entry_idx = t
                earliest_exit_idx = t + off_min
                deadline_idx = min(n - 1, t + off_max)
                n_entries += 1
            elif pending_pos == 0 and pos != 0:
                # Closing current trade
                pos = 0
                entry_idx = None
                earliest_exit_idx = None
                deadline_idx = None
                n_exits += 1
                pnl -= tc  # one round-trip cost per completed trade
            else:
                # Flip directly (not used in this simple design)
                pos = pending_pos
            pending_pos = None

        # Accrue PnL for current bar
        if pos != 0:
            pnl += pos * rFI[t]

        # Exit conditions
        if pos != 0 and entry_idx is not None:
            # Time-based exit: lag window expired or end of day
            if deadline_idx is not None and (t >= deadline_idx or t == n - 1):
                # schedule closing at next bar (or now if last bar)
                if params.enter_on_next_bar and t < n - 1:
                    pending_pos = 0
                else:
                    pos = 0
                    entry_idx = None
                    earliest_exit_idx = None
                    deadline_idx = None
                    n_exits += 1
                    pnl -= tc
            else:
                # Profit-based exit: follower-index cumulative return exceeds r_thresh
                if earliest_exit_idx is not None and t >= earliest_exit_idx:
                    # cumulative follower-index return since entry
                    cum_ret = cum_rFI[t] - (cum_rFI[entry_idx] if entry_idx > 0 else 0.0)
                    # in the direction of current position
                    if pos * cum_ret >= r_thresh:
                        if params.enter_on_next_bar and t < n - 1:
                            pending_pos = 0
                        else:
                            pos = 0
                            entry_idx = None
                            earliest_exit_idx = None
                            deadline_idx = None
                            n_exits += 1
                            pnl -= tc

        # Entry conditions (only if flat and enough history)
        if pos == 0 and entry_idx is None and t >= params.lookback - 1:
            if not np.isfinite(muL[t]) or not np.isfinite(sigL[t]) or sigL[t] <= 0:
                continue

            upper = muL[t] + params.k * sigL[t]
            lower = muL[t] - params.k * sigL[t]
            r_t = rL[t]

            desired_pos = 0
            # Strong positive shock in leader
            if r_t > r_thresh and r_t > upper:
                # follower undervalued: long follower / short index
                desired_pos = +1
            # Strong negative shock in leader
            elif r_t < -r_thresh and r_t < lower:
                # follower overvalued: short follower / long index
                desired_pos = -1

            if desired_pos != 0:
                if params.enter_on_next_bar and t < n - 1:
                    pending_pos = desired_pos
                else:
                    pos = desired_pos
                    entry_idx = t
                    earliest_exit_idx = t + off_min
                    deadline_idx = min(n - 1, t + off_max)
                    n_entries += 1

    # If still in a position at the very end, close it and charge cost
    if pos != 0:
        pos = 0
        n_exits += 1
        pnl -= tc

    return pnl, n_entries, n_exits


# =============================================================================
# Per-day and full backtest
# =============================================================================

def trade_one_day_from_prev_pairs(
    trade_day: pl.Date,
    prev_pairs: pl.DataFrame,
    returns_dir: Path,
    index_ticker: str,
    params: OCPTradeParams,
    returns_col: str = "mid_price_return",
    timestamp_col: str = "timestamp",
) -> Tuple[float, int, int, int]:
    if prev_pairs.height == 0:
        return 0.0, 0, 0, 0

    leaders = prev_pairs["leader"].to_list()
    followers = prev_pairs["follower"].to_list()
    all_tickers = list(dict.fromkeys(leaders + followers + [index_ticker]))

    try:
        timestamps, aligned = load_aligned_returns_for_day(
            returns_dir=returns_dir,
            tickers=all_tickers,
            trade_day=trade_day,
            returns_col=returns_col,
            timestamp_col=timestamp_col,
        )
    except ValueError:
        # No data for this day -> skip
        return 0.0, 0, 0, 0

    has_index = index_ticker in aligned

    total_pnl = 0.0
    pairs_used = 0
    total_entries = 0
    total_exits = 0

    for row in prev_pairs.iter_rows(named=True):
        leader = row["leader"]
        follower = row["follower"]
        l_hat = float(row["l_hat"])
        sigma_l = float(row["sigma_l"]) if "sigma_l" in prev_pairs.columns else 0.0

        if leader not in aligned or follower not in aligned:
            continue

        leader_ret = aligned[leader]
        follower_ret = aligned[follower]
        index_ret = aligned[index_ticker] if has_index else None

        pnl_pair, n_ent, n_ex = trade_one_pair_one_day_ocp_style(
            leader_ret=leader_ret,
            follower_ret=follower_ret,
            index_ret=index_ret,
            lag_hat=l_hat,
            sigma_l=sigma_l,
            params=params,
        )

        if n_ent > 0:
            pairs_used += 1

        total_pnl += pnl_pair
        total_entries += n_ent
        total_exits += n_ex

    return total_pnl, pairs_used, total_entries, total_exits



def run_backtest(
    pairs_path: Path,
    returns_dir: Path,
    index_ticker: str,
    params: Optional[OCPTradeParams] = None,
    returns_col: str = "mid_price_return",
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Run the full OCP-based intraday backtest over all available days.

    The procedure:
      - Read daily top OCP pairs from `pairs_path`.
      - Sort by date and build a rolling formation/trading schedule:
          formation_day = dates[i - 1]
          trade_day     = dates[i]
      - For each (formation_day, trade_day), trade all pairs from formation_day
        on trade_day using `trade_one_day_from_prev_pairs`.
      - Aggregate per-day PnL and basic activity statistics.

    Parameters
    ----------
    pairs_path : Path
        Path to a parquet file containing daily top pairs. The file must have:
          - "date": date/datetime
          - "leader": str
          - "follower": str
          - "l_hat": float
          - "sigma_l": float
        (extra columns are ignored).
    returns_dir : Path
        Directory where per-ticker minute return parquet files are stored.
    index_ticker : str
        Ticker of the market index used as hedge (e.g. "SPX", "SPY").
    params : Optional[OCPTradeParams]
        Strategy parameters; if None, defaults are used.
    returns_col : str
        Name of the return column in per-ticker parquet files.
    timestamp_col : str
        Name of the timestamp column in per-ticker parquet files.

    Returns
    -------
    results : pl.DataFrame
        DataFrame with columns:
          - "trade_day": date
          - "daily_pnl": float
          - "pairs_used": int
          - "entries": int
          - "exits": int
    """
    if params is None:
        params = OCPTradeParams()

    pairs = pl.read_parquet(pairs_path)
    if "date" not in pairs.columns:
        raise ValueError("pairs_path must contain a 'date' column.")

    pairs = pairs.sort("date")
    dates = (
        pairs.select("date")
        .unique()
        .sort("date")["date"]
        .to_list()
    )

    rows: List[Dict] = []

    # Rolling: day[i-1] is formation, day[i] is trading
    for i in range(1, len(dates)):
        formation_day = dates[i - 1]
        trade_day = dates[i]

        prev_pairs = pairs.filter(pl.col("date") == formation_day)

        day_pnl, used, entries, exits = trade_one_day_from_prev_pairs(
            trade_day=trade_day,
            prev_pairs=prev_pairs,
            returns_dir=returns_dir,
            index_ticker=index_ticker,
            params=params,
            returns_col=returns_col,
            timestamp_col=timestamp_col,
        )

        rows.append(
            {
                "trade_day": trade_day,
                "daily_pnl": day_pnl,
                "pairs_used": used,
                "entries": entries,
                "exits": exits,
            }
        )

    return pl.DataFrame(rows)
# =============================================================================