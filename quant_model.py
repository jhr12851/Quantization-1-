"""
Simple quantitative trading model using a moving-average crossover strategy.

The script fetches historical price data (Yahoo Finance, akshare, or Tencent for
A-shares), generates trading signals, runs a vectorized backtest, and prints
performance metrics. It defaults to SPY over the last five years but can be
configured via CLI arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests


def _parse_date(date_str: Optional[str]) -> Optional[dt.datetime]:
    if not date_str:
        return None
    return dt.datetime.strptime(date_str, "%Y-%m-%d")


def _format_tencent_symbol(ticker: str) -> str:
    clean = ticker.strip().lower()
    if not clean:
        raise ValueError("Ticker is empty")
    if "." in clean:
        base, suffix = clean.split(".", 1)
        suffix = suffix.lower()
        if suffix in {"sh", "ss", "sha"}:
            prefix = "sh"
        elif suffix in {"sz", "she", "sza"}:
            prefix = "sz"
        else:
            prefix = "sz" if base.startswith(("0", "3")) else "sh"
        return f"{prefix}{base}"
    prefix = "sh" if clean.startswith(("6", "9")) else "sz"
    return f"{prefix}{clean}"


def compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame(
        {"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist}
    )


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 9,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    lowest_low = low.rolling(window=window, min_periods=1).min()
    highest_high = high.rolling(window=window, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    rsv = ((close - lowest_low) / denom * 100).fillna(0)
    k = rsv.ewm(alpha=1 / smooth_k, adjust=False).mean()
    d = k.ewm(alpha=1 / smooth_d, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def load_price_data(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    *,
    data_source: str = "yahoo",
    auto_adjust: bool = True,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    akshare_adjust: str = "qfq",
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_window: int = 14,
    kdj_window: int = 9,
    kdj_smooth_k: int = 3,
    kdj_smooth_d: int = 3,
) -> pd.DataFrame:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt is None:
        end_dt = dt.datetime.utcnow()
    if start_dt is None:
        start_dt = end_dt - dt.timedelta(days=365 * 5)

    source = data_source.lower()

    if source == "yahoo":
        attempt = 0
        data = pd.DataFrame()
        while attempt <= max_retries:
            data = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=auto_adjust,
            )
            if not data.empty:
                break
            attempt += 1
            if attempt > max_retries:
                break
            pause = retry_delay * attempt
            print(
                f"No data returned for {ticker} (attempt {attempt}/{max_retries}). "
                f"Retrying in {pause:.1f}s..."
            )
            time.sleep(pause)

        if data.empty:
            raise ValueError(
                f"No price data downloaded for {ticker} after {max_retries} retries."
            )

        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        data.index.name = "date"
        data = data.sort_index()
        indicators = [
            compute_macd(data["close"], macd_fast, macd_slow, macd_signal),
            compute_rsi(data["close"], rsi_window).rename("rsi"),
            compute_kdj(
                data["high"],
                data["low"],
                data["close"],
                kdj_window,
                kdj_smooth_k,
                kdj_smooth_d,
            ),
        ]
        data = pd.concat([data, *indicators], axis=1)
        return data

    if source == "akshare":
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "akshare is required when data_source='akshare'. Install via pip install akshare."
            ) from exc

        symbol = ticker.split(".")[0]
        start_str = start_dt.strftime("%Y%m%d")
        # add one day to include end date as akshare uses inclusive boundaries
        end_str = end_dt.strftime("%Y%m%d")

        data = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_str,
            end_date=end_str,
            adjust=None if akshare_adjust.lower() == "none" else akshare_adjust.lower(),
        )
        if data.empty:
            raise ValueError(f"No price data downloaded for {ticker} from akshare.")

        data = data.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        data = data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
        data["adj_close"] = data["close"]
        data = data.sort_index()
        indicators = [
            compute_macd(data["close"], macd_fast, macd_slow, macd_signal),
            compute_rsi(data["close"], rsi_window).rename("rsi"),
            compute_kdj(
                data["high"],
                data["low"],
                data["close"],
                kdj_window,
                kdj_smooth_k,
                kdj_smooth_d,
            ),
        ]
        data = pd.concat([data, *indicators], axis=1)
        return data

    if source == "tencent":
        session = requests.Session()
        session.trust_env = False
        base_url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get"

        symbol_code = _format_tencent_symbol(ticker)
        adjust_norm = akshare_adjust.lower()
        if adjust_norm not in {"qfq", "hfq", "none", "day"}:
            adjust_norm = "qfq"

        if adjust_norm in {"none", "day"}:
            adjust_param = "day"
            dataset_key = "day"
        elif adjust_norm == "hfq":
            adjust_param = "hfq"
            dataset_key = "hfqday"
        else:
            adjust_param = "qfq"
            dataset_key = "qfqday"

        rows: list[dict[str, float | dt.datetime]] = []
        chunk_span = dt.timedelta(days=360)
        current_start = start_dt
        while current_start <= end_dt:
            chunk_end = min(current_start + chunk_span, end_dt)
            param_value = (
                f"{symbol_code},day,{current_start.strftime('%Y-%m-%d')},"
                f"{chunk_end.strftime('%Y-%m-%d')},640,{adjust_param}"
            )
            attempt = 0
            payload = {}
            while attempt <= max_retries:
                try:
                    response = session.get(
                        base_url, params={"param": param_value}, timeout=10
                    )
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    pause = retry_delay * attempt
                    print(
                        f"Tencent data request timed out for {ticker} ({current_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}). "
                        f"Retrying in {pause:.1f}s..."
                    )
                    time.sleep(pause)
                except requests.exceptions.RequestException:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    pause = retry_delay * attempt
                    print(
                        f"Tencent data request failed for {ticker} ({current_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}). "
                        f"Retrying in {pause:.1f}s..."
                    )
                    time.sleep(pause)
            if not payload:
                current_start = chunk_end + dt.timedelta(days=1)
                continue
            symbol_payload = payload.get("data", {}).get(symbol_code, {})
            series = symbol_payload.get(dataset_key, [])
            for entry in series:
                if len(entry) < 6:
                    continue
                date = pd.to_datetime(entry[0], errors="coerce")
                if pd.isna(date):
                    continue
                try:
                    open_px = float(entry[1])
                    close_px = float(entry[2])
                    high_px = float(entry[3])
                    low_px = float(entry[4])
                    volume = float(entry[5])
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "date": date.to_pydatetime(),
                        "open": open_px,
                        "close": close_px,
                        "high": high_px,
                        "low": low_px,
                        "volume": volume,
                    }
                )
            current_start = chunk_end + dt.timedelta(days=1)

        if not rows:
            raise ValueError(f"No price data downloaded for {ticker} from Tencent.")

        data = pd.DataFrame(rows).drop_duplicates(subset="date")
        data = data.set_index("date").sort_index()
        data["adj_close"] = data["close"]
        data = data.loc[(data.index >= start_dt) & (data.index <= end_dt)]
        if data.empty:
            raise ValueError(f"No price data downloaded for {ticker} from Tencent.")
        indicators = [
            compute_macd(data["close"], macd_fast, macd_slow, macd_signal),
            compute_rsi(data["close"], rsi_window).rename("rsi"),
            compute_kdj(
                data["high"],
                data["low"],
                data["close"],
                kdj_window,
                kdj_smooth_k,
                kdj_smooth_d,
            ),
        ]
        data = pd.concat([data, *indicators], axis=1)
        return data

    raise ValueError(
        f"Unsupported data_source '{data_source}'. Use 'yahoo', 'akshare', or 'tencent'."
    )


def moving_average_strategy(
    data: pd.DataFrame,
    short_window: int,
    long_window: int,
    *,
    use_macd: bool = False,
    use_kdj: bool = False,
    use_rsi: bool = False,
    rsi_threshold: float = 50.0,
) -> pd.Series:
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    prices = data["close"]
    short_ma = prices.rolling(window=short_window, min_periods=short_window).mean()
    long_ma = prices.rolling(window=long_window, min_periods=long_window).mean()

    signal = pd.Series(index=prices.index, dtype=float, name="signal")
    signal[short_ma > long_ma] = 1.0
    signal[short_ma <= long_ma] = 0.0
    signal = signal.ffill().fillna(0.0)

    if use_macd:
        if {"macd", "macd_signal"}.issubset(data.columns):
            confirmation = data["macd"] > data["macd_signal"]
            signal = signal.where(confirmation, 0.0)
        else:
            raise ValueError("MACD columns are missing from the price data.")

    if use_kdj:
        if {"kdj_k", "kdj_d"}.issubset(data.columns):
            confirmation = data["kdj_k"] > data["kdj_d"]
            signal = signal.where(confirmation, 0.0)
        else:
            raise ValueError("KDJ columns are missing from the price data.")

    if use_rsi:
        if "rsi" in data.columns:
            confirmation = data["rsi"] > rsi_threshold
            signal = signal.where(confirmation, 0.0)
        else:
            raise ValueError("RSI column is missing from the price data.")

    return signal


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    metrics: dict[str, float]
    risk_flags: pd.Series


def backtest_vectorized(
    data: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    trading_cost_bps: float = 5.0,
) -> BacktestResult:
    if not data.index.equals(signals.index):
        signals = signals.reindex(data.index).ffill().fillna(0.0)

    close = data["close"]
    daily_rets = close.pct_change().fillna(0.0)

    base_positions = signals.shift(1).fillna(0.0)
    drawdown_mask = (daily_rets <= -0.05) & (base_positions > 0)
    positions = base_positions.where(~drawdown_mask, base_positions * 0.5)

    strategy_rets = positions * daily_rets
    trade_diff = positions.diff().fillna(positions).abs()
    trading_cost = trade_diff * (trading_cost_bps / 10_000.0)
    net_rets = strategy_rets - trading_cost

    equity_curve = (1.0 + net_rets).cumprod() * initial_capital
    trades = trade_diff.astype(bool)

    metrics = compute_performance_metrics(net_rets, equity_curve)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_rets,
        positions=positions,
        trades=trades,
        metrics=metrics,
        risk_flags=drawdown_mask,
    )


def compute_performance_metrics(
    returns: pd.Series, equity_curve: pd.Series
) -> dict[str, float]:
    periods_per_year = 252
    cumulative_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    cagr = (1.0 + cumulative_return) ** (
        periods_per_year / max(len(equity_curve) - 1, 1)
    ) - 1.0

    volatility = returns.std() * math.sqrt(periods_per_year)
    sharpe = (
        np.nan
        if volatility == 0.0
        else returns.mean() / returns.std() * math.sqrt(periods_per_year)
    )

    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    max_drawdown = drawdown.min()

    downside = returns[returns < 0].std() * math.sqrt(periods_per_year)
    sortino = (
        np.nan
        if downside == 0.0
        else returns.mean() / returns[returns < 0].std() * math.sqrt(periods_per_year)
    )

    calmar = np.nan if max_drawdown == 0.0 else cagr / abs(max_drawdown)

    metrics = {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": (returns > 0).mean(),
        "avg_trade_return": returns[returns != 0].mean(),
        "num_trades": int(returns[returns != 0].count()),
    }
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    lines = ["Performance metrics:"]
    for key, value in metrics.items():
        if key in {"num_trades"}:
            lines.append(f"  {key:>20}: {value:d}")
        else:
            lines.append(f"  {key:>20}: {value:>8.2%}" if abs(value) < 10 else f"  {key:>20}: {value:>10.4f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    today = dt.datetime.utcnow().date()
    five_years_ago = today - dt.timedelta(days=365 * 5)

    parser = argparse.ArgumentParser(description="Moving-average crossover backtest")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol to backtest")
    parser.add_argument(
        "--start",
        default=str(five_years_ago),
        help="Start date (YYYY-MM-DD). Defaults to five years ago.",
    )
    parser.add_argument(
        "--end",
        default=str(today),
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=20,
        help="Short moving average window",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=100,
        help="Long moving average window",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital for the backtest",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Round-trip trading cost in basis points",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max Yahoo Finance download retries on empty data",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Base delay (seconds) between retries; multiplies by attempt number",
    )
    parser.add_argument(
        "--data-source",
        choices=("yahoo", "akshare", "tencent"),
        default="yahoo",
        help="Price data source",
    )
    parser.add_argument(
        "--auto-adjust",
        dest="auto_adjust",
        action="store_true",
        help="Use Yahoo Finance adjusted prices when data-source=yahoo",
    )
    parser.add_argument(
        "--no-auto-adjust",
        dest="auto_adjust",
        action="store_false",
        help="Disable Yahoo Finance price adjustments",
    )
    parser.set_defaults(auto_adjust=True)
    parser.add_argument(
        "--akshare-adjust",
        default="qfq",
        help="A股数据源复权方式: qfq(前复权)/hfq(后复权)/none(不复权)",
    )
    parser.add_argument(
        "--use-macd",
        action="store_true",
        help="Require MACD line above signal line as additional long confirmation",
    )
    parser.add_argument(
        "--macd-fast",
        type=int,
        default=12,
        help="MACD fast EMA period",
    )
    parser.add_argument(
        "--macd-slow",
        type=int,
        default=26,
        help="MACD slow EMA period",
    )
    parser.add_argument(
        "--macd-signal",
        type=int,
        default=9,
        help="MACD signal EMA period",
    )
    parser.add_argument(
        "--use-kdj",
        action="store_true",
        help="Require KDJ K line above D line",
    )
    parser.add_argument(
        "--kdj-window",
        type=int,
        default=9,
        help="KDJ RSV window",
    )
    parser.add_argument(
        "--kdj-smooth-k",
        type=int,
        default=3,
        help="KDJ K line smoothing factor",
    )
    parser.add_argument(
        "--kdj-smooth-d",
        type=int,
        default=3,
        help="KDJ D line smoothing factor",
    )
    parser.add_argument(
        "--use-rsi",
        action="store_true",
        help="Require RSI above threshold",
    )
    parser.add_argument(
        "--rsi-window",
        type=int,
        default=14,
        help="RSI calculation window",
    )
    parser.add_argument(
        "--rsi-threshold",
        type=float,
        default=50.0,
        help="RSI threshold for long confirmation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_price_data(
        args.ticker,
        args.start,
        args.end,
        data_source=args.data_source,
        auto_adjust=args.auto_adjust,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        akshare_adjust=args.akshare_adjust,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal=args.macd_signal,
        rsi_window=args.rsi_window,
        kdj_window=args.kdj_window,
        kdj_smooth_k=args.kdj_smooth_k,
        kdj_smooth_d=args.kdj_smooth_d,
    )
    signals = moving_average_strategy(
        data,
        args.short_window,
        args.long_window,
        use_macd=args.use_macd,
        use_kdj=args.use_kdj,
        use_rsi=args.use_rsi,
        rsi_threshold=args.rsi_threshold,
    )
    result = backtest_vectorized(
        data,
        signals,
        initial_capital=args.capital,
        trading_cost_bps=args.cost_bps,
    )

    print(format_metrics(result.metrics))
    print("\nLast 5 equity values:")
    print(result.equity_curve.tail())


if __name__ == "__main__":
    main()
