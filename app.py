"""
Finance CSV Dashboard - Total trades, wins, losses only.
Two rows = one trade. Win = Net P&L > 0, Loss = Net P&L < 0. Monthly and yearly.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from analyzer import load_trades_csv, sort_trades, collapse_trades, overall_stats, trades_yearly, trades_monthly, performance_metrics

st.set_page_config(page_title="Finance CSV Analyzer", page_icon="📊", layout="wide")

st.title("📊 Finance CSV Analyzer")
st.caption("Upload CSV → we sort by Trade # (two rows = one trade) and show that sorted file. All analysis uses this sorted data: we use the **Exit row’s** Net P&L per trade — positive = win, negative = loss.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Columns: Trade #, Type, Date and time, Net P&L INR",
    )
    st.divider()
    st.header("Net profit formula")
    st.markdown("**net_profit** = (wins × win value) − (losses × loss value)")
    win_value = st.number_input("Win value (per win)", value=1625, min_value=0, step=1, format="%d")
    loss_value = st.number_input("Loss value (per loss)", value=650, min_value=0, step=1, format="%d")
    st.divider()
    st.header("Capital graph")
    initial_capital = st.number_input("Initial capital", value=20000, min_value=0, step=1000, format="%d", help="Yearly line graph shows this + cumulative net profit over years")
    st.divider()
    st.markdown("**Expected columns**")
    st.markdown("- **Trade #** – trade id (entry + exit = one trade)")
    st.markdown("- **Type** – e.g. Entry short, Exit short")
    st.markdown("- **Date and time**")
    st.markdown("- **Net P&L INR** – positive = win, negative = loss")

if not uploaded:
    st.info("👆 Upload a CSV file in the sidebar to start.")
    st.stop()

try:
    with st.spinner("Loading CSV..."):
        df = load_trades_csv(uploaded)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows after loading.")
    st.stop()

df_sorted = sort_trades(df)
collapsed = collapse_trades(df_sorted)

if collapsed.empty:
    st.warning("Could not collapse trades (each trade should have Entry + Exit rows).")
    st.stop()

# ---- Step 1: Sorted CSV = one row per trade ----
st.subheader("1. Sorted CSV (one row per trade)")
st.markdown("Two rows (Entry + Exit) merged into one. **Total trades** = row count below. Date and time = date + entry time-exit time. Start price = entry, end price = exit. We use the **Exit row’s Net P&L INR** (positive = win, negative = loss).")
display_cols = [c for c in collapsed.columns if not c.startswith("_")]
st.dataframe(collapsed[display_cols], use_container_width=True, hide_index=True)

# ---- Overall stats (not in table) ----
st.subheader("2. Overall stats")
stats = overall_stats(collapsed)
def _fmt(d):
    if d is None or (hasattr(d, "strftime") and pd.isna(d)):
        return "—"
    if hasattr(d, "strftime"):
        return d.strftime("%d-%m-%Y")
    return str(d)

st.markdown(f"**Total trades:** {stats['total_trades']}  \n**Total wins:** {stats['total_wins']}  \n**Total loss:** {stats['total_losses']}  \n**Total breakeven:** {stats.get('total_breakeven', 0)} (Net P&L = 0; not counted as win or loss)")
st.markdown(f"**Longest streak of wins:** {stats['longest_win_streak']} (from {_fmt(stats['longest_win_from'])} to {_fmt(stats['longest_win_to'])})")
st.markdown(f"**Longest streak of loss:** {stats['longest_loss_streak']} (from {_fmt(stats['longest_loss_from'])} to {_fmt(stats['longest_loss_to'])})")

# ---- Step 3: Yearly / Monthly tables ----
st.subheader("3. Yearly — Total trades, Wins, Losses")
yearly = trades_yearly(collapsed, win_value=win_value, loss_value=loss_value)
if yearly.empty:
    st.info("No trade data for yearly summary.")
else:
    totals_row = pd.DataFrame([{
        "year": "Total",
        "total_trades": yearly["total_trades"].sum(),
        "total_wins": yearly["total_wins"].sum(),
        "total_losses": yearly["total_losses"].sum(),
        "profit": yearly["profit"].sum(),
        "loss": yearly["loss"].sum(),
        "net_profit": yearly["net_profit"].sum(),
    }])
    yearly_display = pd.concat([yearly, totals_row], ignore_index=True)
    st.dataframe(yearly_display, use_container_width=True, hide_index=True)
    yearly_plot = yearly.rename(columns={"total_wins": "Wins", "total_losses": "Losses"})
    fig_y = px.bar(
        yearly_plot,
        x="year",
        y=["Wins", "Losses"],
        title="Wins vs Losses by year",
        barmode="group",
        labels={"value": "count"},
    )
    fig_y.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_y, use_container_width=True)
    fig_y_line = px.line(yearly, x="year", y="net_profit", title="Net profit by year")
    fig_y_line.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    fig_y_line.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_y_line, use_container_width=True)
    yearly_sorted = yearly.sort_values("year")
    cumulative_capital = initial_capital + yearly_sorted["net_profit"].cumsum()
    fig_cap = px.line(x=yearly_sorted["year"], y=cumulative_capital, title=f"Capital (initial {initial_capital:,} + cumulative net profit)")
    fig_cap.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Year", yaxis_title="Capital")
    fig_cap.add_hline(y=initial_capital, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_cap, use_container_width=True)

# ---- Monthly: year dropdown ----
st.subheader("4. Monthly — Total trades, Wins, Losses (select year)")
if not yearly.empty:
    year_options = sorted(yearly["year"].astype(int).tolist())
    selected_year = st.selectbox("Select year", options=year_options, index=len(year_options) - 1)
    monthly = trades_monthly(collapsed, year=selected_year, win_value=win_value, loss_value=loss_value)
    if monthly.empty:
        st.info(f"No data for year {selected_year}.")
    else:
        st.dataframe(monthly, use_container_width=True, hide_index=True)
        monthly_plot = monthly.rename(columns={"total_wins": "Wins", "total_losses": "Losses"})
        fig_m = px.bar(
            monthly_plot,
            x="year_month",
            y=["Wins", "Losses"],
            title=f"Wins vs Losses by month — {selected_year}",
            barmode="group",
            labels={"value": "count"},
        )
        fig_m.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), xaxis_tickangle=-45)
        st.plotly_chart(fig_m, use_container_width=True)
        fig_m_line = px.line(monthly, x="year_month", y="net_profit", title=f"Net profit by month — {selected_year}")
        fig_m_line.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), xaxis_tickangle=-45)
        fig_m_line.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_m_line, use_container_width=True)
        cumulative_capital_m = initial_capital + monthly["net_profit"].cumsum()
        fig_cap_m = px.line(x=monthly["year_month"], y=cumulative_capital_m, title=f"Capital (initial {initial_capital:,} + cumulative net profit) — {selected_year}")
        fig_cap_m.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Month", yaxis_title="Capital", xaxis_tickangle=-45)
        fig_cap_m.add_hline(y=initial_capital, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_cap_m, use_container_width=True)
else:
    st.info("No yearly data; cannot show monthly.")

# ---- Performance metrics (Greeks & analysis) ----
st.subheader("5. Performance metrics & ratios")
monthly_all = trades_monthly(collapsed, year=None, win_value=win_value, loss_value=loss_value)
pm = performance_metrics(collapsed, yearly, monthly_all, initial_capital, win_value, loss_value)

def _f(v, decimals=2):
    if isinstance(v, float) and (v == float("inf") or v != v):
        return "—" if v != v else "∞"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

st.markdown("**Trading summary**")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total trades", pm["total_trades"], None)
c2.metric("Total wins", pm["total_wins"], None)
c3.metric("Total losses", pm["total_losses"], None)
c4.metric("Win rate %", _f(pm["win_rate_pct"], 1), None)
c5.metric("Total net profit", f"{pm['total_net_profit']:,.0f}", None)

st.markdown("**Risk & return ratios**")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Sharpe ratio", _f(pm["sharpe_ratio"]), "return per unit volatility")
r2.metric("Sortino ratio", _f(pm["sortino_ratio"]), "return per downside volatility")
r3.metric("Max drawdown %", _f(pm["max_drawdown_pct"]), "largest peak-to-trough decline")
r4.metric("Calmar ratio", _f(pm["calmar_ratio"]), "annual return / max drawdown")
r5.metric("Profit factor", _f(pm["profit_factor"]), "gross profit / gross loss")

st.markdown("**Returns & volatility**")
v1, v2, v3, v4 = st.columns(4)
v1.metric("Avg yearly return %", _f(pm["avg_yearly_return_pct"]), None)
v2.metric("Avg monthly return %", _f(pm["avg_monthly_return_pct"]), None)
v3.metric("Volatility (yearly) %", _f(pm["volatility_yearly_pct"]), None)
v4.metric("Volatility (monthly) %", _f(pm["volatility_monthly_pct"]), None)

st.markdown("**Per-trade**")
e1, e2 = st.columns(2)
e1.metric("Expectancy (avg P&L per trade)", _f(pm["expectancy"]), "from actual trade P&L")

st.divider()
st.caption("Finance CSV Analyzer — no data is stored; analysis runs only on the uploaded file.")
