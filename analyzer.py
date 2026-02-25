"""
Finance CSV Analyzer - Trade-based. No database, in-memory only.
Columns: Trade #, Type, Date and time, Signal, Price INR, Net P&L INR, Cumulative P&L INR.
Two rows = one trade (entry + exit). Per completed trade:
  Win      = Net P&L > 0  (even +0.01 counts as win)
  Loss     = Net P&L < 0  (even -0.01 counts as loss)
  Breakeven = Net P&L == 0 (not counted as win or loss; treated separately)
"""

import pandas as pd
import numpy as np
from typing import Optional

TRADE_COL = "trade_#"
TYPE_COL = "type"
DATE_COL = "date_and_time"
SIGNAL_COL = "signal"
PRICE_COL = "price_inr"
NET_PNL_COL = "net_pnl_inr"
CUMULATIVE_PNL_COL = "cumulative_pnl_inr"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def norm(c):
        c = c.strip().lower().replace(" ", "_").replace("&", "and")
        if "pandl" in c:
            c = c.replace("pandl", "pnl")
        return c
    df.columns = [norm(c) for c in df.columns]
    return df


def _parse_date(ser) -> pd.Series:
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0]
    ser = pd.Series(ser).astype(str)
    out = pd.to_datetime(ser, dayfirst=False, errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(ser, dayfirst=True, errors="coerce")
    return out


def load_trades_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    seen = set()
    col_map = {}
    for c in df.columns:
        if "trade" in c and TRADE_COL not in seen:
            col_map[c] = TRADE_COL
            seen.add(TRADE_COL)
        elif c == "type":
            col_map[c] = TYPE_COL
            seen.add(TYPE_COL)
        elif "date" in c and "time" in c and DATE_COL not in seen:
            col_map[c] = DATE_COL
            seen.add(DATE_COL)
        elif "signal" in c and SIGNAL_COL not in seen:
            col_map[c] = SIGNAL_COL
            seen.add(SIGNAL_COL)
        elif "price" in c and "inr" in c and PRICE_COL not in seen:
            col_map[c] = PRICE_COL
            seen.add(PRICE_COL)
        elif "net" in c and ("pnl" in c or "pandl" in c) and NET_PNL_COL not in seen:
            col_map[c] = NET_PNL_COL
            seen.add(NET_PNL_COL)
        elif "cumulative" in c and ("pnl" in c or "pandl" in c) and CUMULATIVE_PNL_COL not in seen:
            col_map[c] = CUMULATIVE_PNL_COL
            seen.add(CUMULATIVE_PNL_COL)
    df = df.rename(columns=col_map)
    if TRADE_COL not in df.columns:
        df[TRADE_COL] = df.iloc[:, 0]
    if DATE_COL not in df.columns:
        for c in df.columns:
            if "date" in c or "time" in c:
                df[DATE_COL] = df[c]
                break
    if DATE_COL in df.columns and df[DATE_COL].isna().all() and len(df.columns) >= 3:
        df[DATE_COL] = df.iloc[:, 2]
    date_col = df[DATE_COL].iloc[:, 0] if isinstance(df[DATE_COL], pd.DataFrame) else df[DATE_COL]
    df["date"] = _parse_date(date_col)
    df = df.dropna(subset=["date"])
    if NET_PNL_COL in df.columns:
        df[NET_PNL_COL] = pd.to_numeric(df[NET_PNL_COL], errors="coerce").fillna(0)
    else:
        df[NET_PNL_COL] = 0
    if PRICE_COL in df.columns:
        df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    if CUMULATIVE_PNL_COL not in df.columns:
        df[CUMULATIVE_PNL_COL] = np.nan
    else:
        df[CUMULATIVE_PNL_COL] = pd.to_numeric(df[CUMULATIVE_PNL_COL], errors="coerce")
    return df


def sort_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by Trade # then date so entry then exit per trade."""
    df = df.copy()
    if TRADE_COL not in df.columns:
        return df.sort_values("date").reset_index(drop=True)
    df["_type_order"] = (
        df[TYPE_COL].astype(str).str.lower().map({"entry": 0, "entry short": 0, "entry long": 0}).fillna(1)
    )
    df = df.sort_values([TRADE_COL, "date", "_type_order"]).drop(columns=["_type_order"])
    return df.reset_index(drop=True)


def _extract_direction(type_str: str) -> str:
    """From 'Entry short' / 'Exit short' -> 'short'; 'Entry long' / 'Exit long' -> 'long'."""
    t = str(type_str).lower()
    if "short" in t:
        return "short"
    if "long" in t:
        return "long"
    return str(type_str).strip()


def collapse_trades(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Turn two rows per trade (Entry + Exit) into one row.
    Columns: Trade #, Type, Date and time, Signal, start Price, end price, Net P&L INR, Cumulative P&L INR.
    Date and time = date + entry time-exit time. start Price = Entry price, end price = Exit price.
    """
    if TRADE_COL not in df_sorted.columns:
        return pd.DataFrame()
    rows = []
    for trade_id, grp in df_sorted.groupby(TRADE_COL):
        grp = grp.sort_values("date")
        entry_row = grp.iloc[0]
        exit_row = grp.iloc[-1]
        date_exit = exit_row["date"]
        if pd.isna(date_exit):
            continue
        try:
            date_str = date_exit.strftime("%d-%m-%Y")
            entry_time = entry_row["date"].strftime("%H:%M") if pd.notna(entry_row["date"]) else ""
            exit_time = date_exit.strftime("%H:%M")
            date_and_time_str = f"{date_str} {entry_time}-{exit_time}"
        except Exception:
            date_and_time_str = str(exit_row.get(DATE_COL, ""))
        type_val = _extract_direction(entry_row.get(TYPE_COL, ""))
        signal_val = exit_row.get(SIGNAL_COL, "") if SIGNAL_COL in grp.columns else ""
        start_price = entry_row.get(PRICE_COL, np.nan)
        end_price = exit_row.get(PRICE_COL, np.nan)
        net_pnl = exit_row.get(NET_PNL_COL, 0)
        cum_pnl = exit_row.get(CUMULATIVE_PNL_COL, np.nan)
        _net_pnl_val = pd.to_numeric(net_pnl, errors="coerce")
        _net_pnl_val = 0.0 if pd.isna(_net_pnl_val) else float(_net_pnl_val)
        rows.append({
            "Trade #": trade_id,
            "Type": type_val,
            "Date and time": date_and_time_str,
            "Signal": signal_val,
            "start Price": start_price,
            "end price": end_price,
            "Net P&L INR": net_pnl,
            "Cumulative P&L INR": cum_pnl,
            "_date": date_exit,
            "_net_pnl": _net_pnl_val,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Win = Net P&L > 0, Loss = Net P&L < 0, Breakeven = 0 (not counted as win or loss)
    out["_win"] = out["_net_pnl"] > 0
    out["_loss"] = out["_net_pnl"] < 0
    out["_breakeven"] = out["_net_pnl"] == 0
    out["W&L"] = out["_net_pnl"].apply(lambda x: 1 if x > 0 else (0 if x < 0 else "BE"))
    return out


def _find_streaks(collapsed_df: pd.DataFrame) -> dict:
    """Longest win streak and longest loss streak with date ranges. Trades in date order."""
    if collapsed_df.empty or "_date" not in collapsed_df.columns or "_win" not in collapsed_df.columns:
        return {
            "total_trades": 0, "total_wins": 0, "total_losses": 0, "total_breakeven": 0,
            "longest_win_streak": 0, "longest_win_from": None, "longest_win_to": None,
            "longest_loss_streak": 0, "longest_loss_from": None, "longest_loss_to": None,
        }
    df = collapsed_df.sort_values("_date").reset_index(drop=True)
    total_trades = len(df)
    total_wins = int(df["_win"].sum())
    total_losses = int(df["_loss"].sum())
    total_breakeven = int(df["_breakeven"].sum()) if "_breakeven" in df.columns else 0
    # Longest win streak
    best_win_len, best_win_start, best_win_end = 0, None, None
    cur_start, cur_len = None, 0
    for i in range(len(df)):
        if df.iloc[i]["_win"]:
            if cur_start is None:
                cur_start = df.iloc[i]["_date"]
            cur_len += 1
        else:
            if cur_len > best_win_len:
                best_win_len, best_win_start = cur_len, cur_start
                best_win_end = df.iloc[i - 1]["_date"] if i > 0 else cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_win_len:
        best_win_len, best_win_start = cur_len, cur_start
        best_win_end = df.iloc[-1]["_date"]
    # Longest loss streak
    best_loss_len, best_loss_start, best_loss_end = 0, None, None
    cur_start, cur_len = None, 0
    for i in range(len(df)):
        if df.iloc[i]["_loss"]:
            if cur_start is None:
                cur_start = df.iloc[i]["_date"]
            cur_len += 1
        else:
            if cur_len > best_loss_len:
                best_loss_len, best_loss_start = cur_len, cur_start
                best_loss_end = df.iloc[i - 1]["_date"] if i > 0 else cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_loss_len:
        best_loss_len, best_loss_start = cur_len, cur_start
        best_loss_end = df.iloc[-1]["_date"]
    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_breakeven": total_breakeven,
        "longest_win_streak": best_win_len,
        "longest_win_from": best_win_start,
        "longest_win_to": best_win_end,
        "longest_loss_streak": best_loss_len,
        "longest_loss_from": best_loss_start,
        "longest_loss_to": best_loss_end,
    }


def overall_stats(collapsed_df: pd.DataFrame) -> dict:
    """Overall stats including longest win/loss streaks with date ranges."""
    return _find_streaks(collapsed_df)


def trades_yearly(
    collapsed_df: pd.DataFrame,
    win_value: float = 1625,
    loss_value: float = 650,
) -> pd.DataFrame:
    """Total trades, total wins, total losses, net_profit per year. net_profit = (wins * win_value) - (losses * loss_value)."""
    if collapsed_df.empty or "_date" not in collapsed_df.columns:
        return pd.DataFrame(columns=["year", "total_trades", "total_wins", "total_losses", "net_profit"])
    df = collapsed_df.copy()
    df["year"] = df["_date"].dt.year
    out = df.groupby("year").agg(
        total_trades=("Trade #", "count"),
        total_wins=("_win", "sum"),
        total_losses=("_loss", "sum"),
    ).reset_index()
    out["total_wins"] = out["total_wins"].astype(int)
    out["total_losses"] = out["total_losses"].astype(int)
    out["profit"] = out["total_wins"] * win_value
    out["loss"] = out["total_losses"] * loss_value
    out["net_profit"] = out["profit"] - out["loss"]
    return out


def trades_monthly(
    collapsed_df: pd.DataFrame,
    year: Optional[int] = None,
    win_value: float = 1625,
    loss_value: float = 650,
) -> pd.DataFrame:
    """Total trades, total wins, total losses, net_profit per month. net_profit = (wins * win_value) - (losses * loss_value). If year set, only that year."""
    if collapsed_df.empty or "_date" not in collapsed_df.columns:
        return pd.DataFrame(columns=["year_month", "total_trades", "total_wins", "total_losses", "net_profit"])
    df = collapsed_df.copy()
    df["year_month"] = df["_date"].dt.to_period("M").astype(str)
    if year is not None:
        df = df[df["_date"].dt.year == year]
    if df.empty:
        return pd.DataFrame(columns=["year_month", "total_trades", "total_wins", "total_losses", "net_profit"])
    out = df.groupby("year_month").agg(
        total_trades=("Trade #", "count"),
        total_wins=("_win", "sum"),
        total_losses=("_loss", "sum"),
    ).reset_index()
    out["total_wins"] = out["total_wins"].astype(int)
    out["total_losses"] = out["total_losses"].astype(int)
    out["profit"] = out["total_wins"] * win_value
    out["loss"] = out["total_losses"] * loss_value
    out["net_profit"] = out["profit"] - out["loss"]
    return out


def performance_metrics(
    collapsed_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    monthly_all_df: pd.DataFrame,
    initial_capital: float,
    win_value: float,
    loss_value: float,
) -> dict:
    """
    All metrics derived from yearly net profit (formula-based).
    Sharpe, Sortino, drawdown, profit factor, returns, volatility, etc.
    """
    out = {
        "total_trades": 0, "total_wins": 0, "total_losses": 0, "win_rate_pct": 0.0,
        "profit_factor": 0.0, "expectancy": 0.0, "total_net_profit": 0.0,
        "avg_yearly_return_pct": 0.0, "avg_monthly_return_pct": 0.0,
        "volatility_yearly_pct": 0.0, "volatility_monthly_pct": 0.0,
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0, "calmar_ratio": 0.0,
    }
    if collapsed_df.empty:
        return out
    n = len(collapsed_df)
    out["total_trades"] = n
    out["total_wins"] = int(collapsed_df["_win"].sum())
    out["total_losses"] = int(collapsed_df["_loss"].sum())
    out["win_rate_pct"] = (out["total_wins"] / n * 100) if n else 0
    out["total_net_profit"] = float(yearly_df["net_profit"].sum()) if not yearly_df.empty else 0

    if not yearly_df.empty and len(yearly_df) >= 1:
        ys = yearly_df.sort_values("year").reset_index(drop=True)
        # Profit factor = gross profit / gross loss (from yearly profit & loss columns)
        gross_profit = float(ys["profit"].sum())
        gross_loss = float(ys["loss"].sum())
        if gross_loss > 0:
            out["profit_factor"] = gross_profit / gross_loss
        else:
            out["profit_factor"] = float("inf") if gross_profit > 0 else 0
        out["expectancy"] = float(np.mean(ys["net_profit"].values))

        # Avg yearly return % = (avg yearly net profit / initial capital) * 100
        out["avg_yearly_return_pct"] = (float(np.mean(ys["net_profit"])) / initial_capital * 100) if initial_capital else 0

        cap = initial_capital + ys["net_profit"].cumsum()
        cap_prev = np.concatenate([[initial_capital], cap.values[:-1]])
        yearly_ret = (cap.values - cap_prev) / cap_prev
        vol_y = np.std(yearly_ret)
        out["volatility_yearly_pct"] = float(vol_y) * 100
        if vol_y > 0:
            out["sharpe_ratio"] = float(np.mean(yearly_ret) / vol_y)
        dr_ret = yearly_ret[yearly_ret < 0]
        if len(dr_ret) > 0:
            dd_std = np.std(dr_ret)
            if dd_std > 0:
                out["sortino_ratio"] = float(np.mean(yearly_ret) / dd_std)
        peak = np.maximum.accumulate(cap.values)
        dd = (peak - cap.values) / np.where(peak > 0, peak, 1)
        out["max_drawdown_pct"] = float(np.max(dd)) * 100
        if out["max_drawdown_pct"] > 0:
            out["calmar_ratio"] = (out["avg_yearly_return_pct"] / 100) / (out["max_drawdown_pct"] / 100)
        # Avg monthly return % from initial capital (use monthly data if available, else yearly/12)
        if not monthly_all_df.empty and len(monthly_all_df) >= 1 and initial_capital:
            out["avg_monthly_return_pct"] = float(monthly_all_df["net_profit"].mean()) / initial_capital * 100
        else:
            out["avg_monthly_return_pct"] = out["avg_yearly_return_pct"] / 12
        out["volatility_monthly_pct"] = out["volatility_yearly_pct"] / (12 ** 0.5)
    return out
