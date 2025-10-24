"""
Flask web application exposing the quantitative trading model and rendering a
front-end dashboard for metrics and equity curve visualization.
"""

from __future__ import annotations

import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from requests import exceptions as requests_exceptions

from quant_model import (
    backtest_vectorized,
    load_price_data,
    moving_average_strategy,
)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["JSON_SORT_KEYS"] = False

EXECUTOR = ThreadPoolExecutor(max_workers=4)
BACKTEST_TIMEOUT_SECONDS = 60


def _default_dates() -> tuple[str, str]:
    today = dt.datetime.utcnow().date()
    five_years_ago = today - dt.timedelta(days=365 * 5)
    return str(five_years_ago), str(today)


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    try:
        import numpy as np  # lazy import to avoid hard dependency here

        if isinstance(value, (np.generic,)):
            value = value.item()
    except ImportError:
        pass

    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    if isinstance(value, int):
        return value

    return value


@app.route("/")
def index() -> Any:
    return app.send_static_file("index.html")


@app.route("/api/backtest", methods=["POST"])
def api_backtest() -> Any:
    payload = request.get_json(silent=True) or {}

    start_default, end_default = _default_dates()

    ticker = payload.get("ticker", "SPY")
    start = payload.get("start", start_default)
    end = payload.get("end", end_default)
    short_window = int(payload.get("shortWindow", 20))
    long_window = int(payload.get("longWindow", 100))
    capital = float(payload.get("capital", 100_000.0))
    cost_bps = float(payload.get("costBps", 5.0))
    data_source = payload.get("dataSource", "yahoo")
    auto_adjust = bool(payload.get("autoAdjust", True))
    akshare_adjust = payload.get("akshareAdjust", "qfq")
    max_retries = int(payload.get("maxRetries", 3))
    retry_delay = float(payload.get("retryDelay", 5.0))
    use_macd = bool(payload.get("useMacd", False))
    macd_fast = int(payload.get("macdFast", 12))
    macd_slow = int(payload.get("macdSlow", 26))
    macd_signal = int(payload.get("macdSignal", 9))
    use_kdj = bool(payload.get("useKdj", False))
    kdj_window = int(payload.get("kdjWindow", 9))
    kdj_smooth_k = int(payload.get("kdjSmoothK", 3))
    kdj_smooth_d = int(payload.get("kdjSmoothD", 3))
    use_rsi = bool(payload.get("useRsi", False))
    rsi_window = int(payload.get("rsiWindow", 14))
    rsi_threshold = float(payload.get("rsiThreshold", 50.0))

    def _compute() -> Dict[str, Any]:
        data = load_price_data(
            ticker,
            start,
            end,
            data_source=data_source,
            auto_adjust=auto_adjust,
            max_retries=max_retries,
            retry_delay=retry_delay,
            akshare_adjust=akshare_adjust,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            rsi_window=rsi_window,
            kdj_window=kdj_window,
            kdj_smooth_k=kdj_smooth_k,
            kdj_smooth_d=kdj_smooth_d,
        )
        signals = moving_average_strategy(
            data,
            short_window,
            long_window,
            use_macd=use_macd,
            use_kdj=use_kdj,
            use_rsi=use_rsi,
            rsi_threshold=rsi_threshold,
        )
        result = backtest_vectorized(
            data,
            signals,
            initial_capital=capital,
            trading_cost_bps=cost_bps,
        )

        equity_curve_series = result.equity_curve
        equity_curve = [
            {"date": idx.strftime("%Y-%m-%d"), "value": _to_serializable(val)}
            for idx, val in equity_curve_series.items()
        ]
        returns = [
            {"date": idx.strftime("%Y-%m-%d"), "value": _to_serializable(val)}
            for idx, val in result.returns.items()
        ]
        daily_df = data[["close"]].copy()
        daily_df["position"] = result.positions
        daily_df["signal"] = signals
        daily_df["return"] = result.returns
        signal_diff = signals.diff().fillna(signals)
        daily_df["action"] = signal_diff.apply(
            lambda x: "买入" if x > 0 else ("卖出" if x < 0 else "-")
        )
        if {"macd", "macd_signal", "macd_hist"}.issubset(data.columns):
            daily_df["macd"] = data["macd"]
            daily_df["macd_signal"] = data["macd_signal"]
            daily_df["macd_hist"] = data["macd_hist"]
        if {"kdj_k", "kdj_d", "kdj_j"}.issubset(data.columns):
            daily_df["kdj_k"] = data["kdj_k"]
            daily_df["kdj_d"] = data["kdj_d"]
            daily_df["kdj_j"] = data["kdj_j"]
        if "rsi" in data.columns:
            daily_df["rsi"] = data["rsi"]

        close_series = data["close"]
        equity_aligned = equity_curve_series.reindex(data.index).ffill().bfill()
        base_equity = equity_aligned.iloc[0] if not equity_aligned.empty else capital
        if not base_equity:
            base_equity = capital or 1.0
        total_return_series = equity_aligned / base_equity - 1.0
        holding_values: list[float] = []
        current_cost = np.nan
        for idx in data.index:
            change = signal_diff.loc[idx]
            if change > 0:
                current_cost = close_series.loc[idx]
            elif change < 0 and signals.loc[idx] <= 0:
                current_cost = np.nan
            holding_values.append(current_cost if signals.loc[idx] > 0 else np.nan)

        holding_series = pd.Series(holding_values, index=data.index)
        daily_df["holding_price"] = holding_series
        daily_df["total_return"] = total_return_series
        daily_df["risk_reduced"] = result.risk_flags
        daily_data = []
        for idx, row in daily_df.iterrows():
            holding_raw = row.get("holding_price")
            holding_val = None if pd.isna(holding_raw) else _to_serializable(holding_raw)
            daily_data.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": _to_serializable(row["close"]),
                    "position": _to_serializable(row["position"]),
                    "holdingPrice": holding_val,
                    "action": row["action"],
                    "return": _to_serializable(row["return"]),
                    "totalReturn": _to_serializable(row.get("total_return")),
                    "macd": _to_serializable(row.get("macd")),
                    "macdSignal": _to_serializable(row.get("macd_signal")),
                    "macdHist": _to_serializable(row.get("macd_hist")),
                    "kdjK": _to_serializable(row.get("kdj_k")),
                    "kdjD": _to_serializable(row.get("kdj_d")),
                    "kdjJ": _to_serializable(row.get("kdj_j")),
                    "rsi": _to_serializable(row.get("rsi")),
                    "riskReduced": bool(row.get("risk_reduced", False)),
                }
            )
        tail_df = daily_df.tail(10)
        tail_data = []
        for idx, row in tail_df.iterrows():
            holding_raw = row.get("holding_price")
            holding_val = None if pd.isna(holding_raw) else _to_serializable(holding_raw)
            tail_data.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": _to_serializable(row["close"]),
                    "position": _to_serializable(row["position"]),
                    "holdingPrice": holding_val,
                    "action": row["action"],
                    "return": _to_serializable(row["return"]),
                    "totalReturn": _to_serializable(row.get("total_return")),
                    "macd": _to_serializable(row.get("macd")),
                    "macdSignal": _to_serializable(row.get("macd_signal")),
                    "macdHist": _to_serializable(row.get("macd_hist")),
                    "kdjK": _to_serializable(row.get("kdj_k")),
                    "kdjD": _to_serializable(row.get("kdj_d")),
                    "kdjJ": _to_serializable(row.get("kdj_j")),
                    "rsi": _to_serializable(row.get("rsi")),
                    "riskReduced": bool(row.get("risk_reduced", False)),
                }
            )

        ohlc_data = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": _to_serializable(row["open"]),
                "high": _to_serializable(row["high"]),
                "low": _to_serializable(row["low"]),
                "close": _to_serializable(row["close"]),
            }
            for idx, row in data.iterrows()
        ]

        metrics: Dict[str, Any] = {
            key: _to_serializable(value) for key, value in result.metrics.items()
        }
        return {
            "success": True,
            "params": {
                "ticker": ticker,
                "start": start,
                "end": end,
                "shortWindow": short_window,
                "longWindow": long_window,
                "capital": capital,
                "costBps": cost_bps,
                "dataSource": data_source,
                "autoAdjust": auto_adjust,
                "akshareAdjust": akshare_adjust,
                "useMacd": use_macd,
                "macdFast": macd_fast,
                "macdSlow": macd_slow,
                "macdSignal": macd_signal,
                "useKdj": use_kdj,
                "kdjWindow": kdj_window,
                "kdjSmoothK": kdj_smooth_k,
                "kdjSmoothD": kdj_smooth_d,
                "useRsi": use_rsi,
                "rsiWindow": rsi_window,
                "rsiThreshold": rsi_threshold,
            },
            "metrics": metrics,
            "equityCurve": equity_curve,
            "returns": returns,
            "tail": tail_data,
            "daily": daily_data,
            "ohlc": ohlc_data,
        }

    future = EXECUTOR.submit(_compute)
    try:
        response = future.result(timeout=BACKTEST_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        return jsonify(
            {
                "success": False,
                "error": "回测计算超时，请尝试缩短时间范围、切换数据源或稍后重试。",
            }
        )
    except requests_exceptions.ProxyError:
        return jsonify(
            {
                "success": False,
                "error": "无法连接行情代理，请检查网络或取消系统代理后重试。",
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"success": False, "error": str(exc)})

    return jsonify(response)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    port = int((Path(".") / "config.port").read_text().strip()) if Path("config.port").exists() else 5000
    app.run(debug=True, port=port)
