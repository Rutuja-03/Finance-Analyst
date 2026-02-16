"""
Finance CSV Dashboard - Total trades, wins, losses only.
Two rows = one trade. Win = Net P&L > 0, Loss = Net P&L < 0. Monthly and yearly.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from analyzer import load_trades_csv, sort_trades, collapse_trades, overall_stats, trades_yearly, trades_monthly

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

st.markdown(f"**Total trades:** {stats['total_trades']}  \n**Total wins:** {stats['total_wins']}  \n**Total loss:** {stats['total_losses']}")
st.markdown(f"**Longest streak of wins:** {stats['longest_win_streak']} (from {_fmt(stats['longest_win_from'])} to {_fmt(stats['longest_win_to'])})")
st.markdown(f"**Longest streak of loss:** {stats['longest_loss_streak']} (from {_fmt(stats['longest_loss_from'])} to {_fmt(stats['longest_loss_to'])})")

# ---- Step 3: Yearly / Monthly tables ----
st.subheader("3. Yearly — Total trades, Wins, Losses")
yearly = trades_yearly(collapsed)
if yearly.empty:
    st.info("No trade data for yearly summary.")
else:
    st.dataframe(yearly, use_container_width=True, hide_index=True)
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

# ---- Monthly: year dropdown ----
st.subheader("4. Monthly — Total trades, Wins, Losses (select year)")
if not yearly.empty:
    year_options = sorted(yearly["year"].astype(int).tolist())
    selected_year = st.selectbox("Select year", options=year_options, index=len(year_options) - 1)
    monthly = trades_monthly(collapsed, year=selected_year)
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
else:
    st.info("No yearly data; cannot show monthly.")

st.divider()
st.caption("Finance CSV Analyzer — no data is stored; analysis runs only on the uploaded file.")
