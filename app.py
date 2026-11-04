"""Transaction Analytics and Preprocessing Dashboard.

Run:  streamlit run analytics_app.py
Upload a CSV with: TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT
"""
import html
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transaction Analytics & Fraud Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1E222D; border: 1px solid #2E3440; border-radius: 10px;
        padding: 15px 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label { color: #A0AEC0 !important; font-size: 0.85rem !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ECEFF4 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    .main-header {
        background: linear-gradient(90deg, #1A1C24 0%, #2D3748 100%); padding: 24px; border-radius: 12px;
        margin-bottom: 24px; border-left: 6px solid #4C566A;
    }
    .main-header h1 { color: #FFFFFF; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p { color: #CBD5E0; margin: 6px 0 0 0; font-size: 1rem; }
    .kpi-row { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 14px; }
    .kpi { flex: 1 1 12rem; border-radius: 12px; padding: 14px 18px; border: 1px solid #2E3440; border-left: 5px solid var(--c);
           background: linear-gradient(135deg, var(--g) 0%, #1E222D 70%);
           box-shadow: 0 4px 10px rgba(0,0,0,.3); }
    .kpi .l { color: #A0AEC0; font-size: .82rem; font-weight: 600; }
    .kpi .v { color: #FFFFFF; font-size: clamp(1.25rem, 1.9vw, 1.65rem); font-weight: 700; line-height: 1.25; white-space: nowrap; }
    .kpi .s { color: var(--c); font-size: .8rem; font-weight: 600; margin-top: 2px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 8px; padding-left: 18px; padding-right: 18px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA PROCESSING & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = {"TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"}
PREVIEW_ROWS = 1000


def rolling_count(group: pd.DataFrame, window: str) -> pd.Series:
    """Count earlier transactions in the window. The current row is excluded (closed='left').

    The result is rebuilt on the group's own unique row labels, which avoids the
    'cannot reindex on an axis with duplicate labels' error that occurs when many
    transactions share the same timestamp.
    """
    s = group.set_index("TX_DATETIME")["TRANSACTION_ID"]
    counts = s.rolling(window, closed="left").count()
    return pd.Series(counts.values, index=group.index)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the schema, clean the data, and add analytical features."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Coerce types; bad values become NaN and are dropped below instead of crashing
    df["TRANSACTION_ID"] = pd.to_numeric(df["TRANSACTION_ID"], errors="coerce")
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"], errors="coerce")
    df["TX_AMOUNT"] = pd.to_numeric(df["TX_AMOUNT"], errors="coerce")
    df = df.dropna(subset=["TRANSACTION_ID", "TX_DATETIME", "TX_AMOUNT", "CUSTOMER_ID", "TERMINAL_ID"])
    df["TRANSACTION_ID"] = df["TRANSACTION_ID"].astype(np.int64)
    df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype(str)
    df["TERMINAL_ID"] = df["TERMINAL_ID"].astype(str)
    df["TX_AMOUNT"] = df["TX_AMOUNT"].astype(np.float32)
    if "TX_FRAUD" in df.columns:  # optional label column: 1 = fraud, 0 = legitimate; unreadable values count as 0
        df["TX_FRAUD"] = pd.to_numeric(df["TX_FRAUD"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    df = df[df["TX_AMOUNT"] > 0]
    df = df.drop_duplicates(subset=["TRANSACTION_ID"])
    df = df.sort_values("TX_DATETIME", kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows left after cleaning (check dates, amounts and IDs).")

    # Temporal features
    df["TX_DURING_WEEKEND"] = df["TX_DATETIME"].dt.dayofweek.isin([5, 6]).astype(int)
    df["TX_DURING_NIGHT"] = df["TX_DATETIME"].dt.hour.isin(range(0, 7)).astype(int)
    df["TX_PERIOD"] = np.where(df["TX_DURING_NIGHT"] == 1, "Night (00-06)", "Day (07-23)")
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.day_name()

    # Terminal activity: earlier transactions at the same terminal in the last 1 and 7 days
    cols = ["TX_DATETIME", "TRANSACTION_ID"]
    grouped = df.groupby("TERMINAL_ID", group_keys=False)[cols]
    df["NB_TX_TERMINAL_1D"] = grouped.apply(lambda g: rolling_count(g, "1D")).fillna(0)
    df["NB_TX_TERMINAL_7D"] = grouped.apply(lambda g: rolling_count(g, "7D")).fillna(0)
    return df


@st.cache_data(show_spinner="Reading and preprocessing the dataset...")
def get_data(file_key: str, _file_bytes: bytes) -> pd.DataFrame:
    """Cached on file_key (name and size) only, so the large DataFrame is not re-hashed on every rerun."""
    return preprocess(pd.read_csv(io.BytesIO(_file_bytes)))


@st.cache_data(show_spinner=False)
def filtered_csv(file_key: str, _file_bytes: bytes, d0, d1, a0: float, a1: float) -> bytes:
    """CSV bytes for the current filters, cached so the download button does not rebuild them every rerun."""
    return apply_filters(get_data(file_key, _file_bytes), d0, d1, a0, a1).to_csv(index=False).encode("utf-8")


COLORS = {"blue": "#3B82F6", "green": "#10B981", "orange": "#F59E0B", "red": "#EF4444", "purple": "#A855F7"}


def kpi_row(items):
    """Colored KPI cards. Each card's minimum width follows its text length, and the row wraps instead of truncating."""
    cards = []
    for label, value, sub, color, tip in items:
        min_rem = max(10.0, len(label) * 0.5 + 2.5, len(value) * 0.95 + 2.5, len(sub) * 0.45 + 2.5)
        grow = max(1.0, len(value) / 8)
        cards.append(
            f'<div class="kpi" style="--c:{COLORS[color]};--g:{COLORS[color]}38;flex:{grow:.2f} 1 {min_rem:.1f}rem" title="{html.escape(tip, quote=True)}">'
            f'<div class="l">{html.escape(label)}</div><div class="v">{html.escape(value)}</div>'
            f'<div class="s">{html.escape(sub)}</div></div>')
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def fraud_csv(file_key: str, _file_bytes: bytes, d0, d1, a0: float, a1: float) -> bytes:
    """All fraud rows (with transaction IDs) for the current filters."""
    f = apply_filters(get_data(file_key, _file_bytes), d0, d1, a0, a1)
    cols = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]
    return f.loc[f["TX_FRAUD"] == 1, cols].to_csv(index=False).encode("utf-8")


def apply_filters(data: pd.DataFrame, d0, d1, a0: float, a1: float) -> pd.DataFrame:
    day = data["TX_DATETIME"].dt.date
    mask = (day >= d0) & (day <= d1) & (data["TX_AMOUNT"] >= a0) & (data["TX_AMOUNT"] <= a1)
    return data[mask]


# -----------------------------------------------------------------------------
# 3. USER INTERFACE & APP WORKFLOW
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <h1>💳 Transaction Analytics & Preprocessing Dashboard</h1>
        <p>Automated feature engineering, fraud-risk profiling, and transaction exploration engine.</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Raw CSV", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a transaction CSV file from the sidebar to begin.")
    st.stop()

try:
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"
    file_bytes = uploaded_file.getvalue()
    data = get_data(file_key, file_bytes)
except ValueError as err:
    st.error(f"❌ {err}")
    st.stop()
except Exception as err:
    st.error("Could not read or process this file.")
    st.exception(err)
    st.stop()

# Sidebar filters
st.sidebar.divider()
st.sidebar.header("🔍 Global Filters")
min_d, max_d = data["TX_DATETIME"].min().date(), data["TX_DATETIME"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if not (isinstance(date_range, (tuple, list)) and len(date_range) == 2):
    st.warning("Pick both a start date and an end date to apply the date filter.")
    st.stop()
d0, d1 = date_range

amt_min, amt_max = float(data["TX_AMOUNT"].min()), float(data["TX_AMOUNT"].max())
if amt_min < amt_max:
    a0, a1 = st.sidebar.slider("Amount ($)", amt_min, amt_max, (amt_min, amt_max))
else:
    a0, a1 = amt_min, amt_max
    st.sidebar.caption(f"All amounts equal ${amt_min:,.2f}; amount filter disabled.")

filtered_df = apply_filters(data, d0, d1, a0, a1)
if filtered_df.empty:
    st.warning("No transactions match the current filters. Widen the date range or amount range.")
    st.stop()

# KPI bar (amounts summed in float64 so totals stay accurate)
amounts = filtered_df["TX_AMOUNT"].astype("float64")
n = len(filtered_df)
night_n, wknd_n = int(filtered_df["TX_DURING_NIGHT"].sum()), int(filtered_df["TX_DURING_WEEKEND"].sum())
kpi_row([
    ("Total Transactions", f"{n:,}", "in the current filters", "blue", "Number of transactions after cleaning and filtering."),
    ("Total Volume", f"${amounts.sum():,.2f}", "sum of all amounts", "green", "Total value of all transactions in the current filters."),
    ("Avg Amount", f"${amounts.mean():,.2f}", "per transaction", "purple", "Mean transaction amount."),
    ("Night Transactions", f"{night_n:,}", f"{night_n / n * 100:.1f}% of all", "orange", "Transactions between 00:00 and 06:59."),
    ("Weekend Transactions", f"{wknd_n:,}", f"{wknd_n / n * 100:.1f}% of all", "blue", "Transactions on Saturday or Sunday."),
])

has_fraud = "TX_FRAUD" in filtered_df.columns
if has_fraud:
    fraud_df = filtered_df[filtered_df["TX_FRAUD"] == 1]
    nf = len(fraud_df)
    fraud_amt = fraud_df["TX_AMOUNT"].astype("float64").sum()
    kpi_row([
        ("Fraud Transactions", f"{nf:,}", "rows with TX_FRAUD = 1", "red", "Rows with TX_FRAUD = 1 in the current filters."),
        ("Fraud Rate", f"{nf / n * 100:.2f}%", "of all transactions", "orange", "Fraud transactions divided by all transactions in the current filters."),
        ("Fraud Amount", f"${fraud_amt:,.2f}", "total value of fraud", "purple", "Total value of the fraudulent transactions."),
        ("Share of Volume", f"{fraud_amt / amounts.sum() * 100:.2f}%", "of total volume", "blue", "Fraud amount divided by total volume."),
    ])
else:
    st.info("Add a TX_FRAUD column (1 = fraud, 0 = legitimate) to your CSV to see fraud counts and transaction IDs.")

st.divider()
tab_names = ["📊 Analytics & Trends", "📋 Data Explorer", "⚡ Feature Matrix"] + (["🚨 Fraud Transactions"] if has_fraud else [])
tabs = st.tabs(tab_names)
tab1, tab2, tab3 = tabs[:3]

# TAB 1: VISUAL ANALYTICS
with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        daily_vol = filtered_df.groupby(filtered_df["TX_DATETIME"].dt.date)["TX_AMOUNT"].sum().reset_index()
        daily_vol["TX_AMOUNT"] = daily_vol["TX_AMOUNT"].astype("float64")
        fig_vol = px.line(daily_vol, x="TX_DATETIME", y="TX_AMOUNT", title="<b>Daily Transaction Volume ($)</b>",
                          labels={"TX_DATETIME": "Date", "TX_AMOUNT": "Total Volume ($)"}, template="plotly_dark")
        fig_vol.update_traces(line_color="#8884d8", line_width=2)
        st.plotly_chart(fig_vol, use_container_width=True)
    with col_right:
        hourly = filtered_df.groupby("TX_HOUR")["TRANSACTION_ID"].count().reset_index()
        fig_hour = px.bar(hourly, x="TX_HOUR", y="TRANSACTION_ID", title="<b>Hourly Transaction Velocity</b>",
                          labels={"TX_HOUR": "Hour of Day (0-23)", "TRANSACTION_ID": "Transaction Count"}, template="plotly_dark")
        fig_hour.update_traces(marker_color="#82ca9d")
        st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Terminal Transaction Velocity Profile")
    sample = filtered_df.sample(min(5000, n), random_state=42)
    fig_scatter = px.scatter(
        sample, x="NB_TX_TERMINAL_1D", y="NB_TX_TERMINAL_7D", size="TX_AMOUNT", color="TX_PERIOD",
        hover_data=["TERMINAL_ID", "CUSTOMER_ID"],
        title=f"<b>1-Day vs 7-Day Terminal Activity (Sampled {len(sample):,} Rows)</b>",
        labels={"NB_TX_TERMINAL_1D": "Earlier terminal transactions, last 1 day",
                "NB_TX_TERMINAL_7D": "Earlier terminal transactions, last 7 days", "TX_PERIOD": "Time of day"},
        template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 2: DATA TABLE & EXPORT
with tab2:
    st.subheader("Filtered Dataset View")
    st.caption(f"Showing the first {min(PREVIEW_ROWS, n):,} of {n:,} rows. The download contains all rows.")
    explorer_cols = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"] + (["TX_FRAUD"] if has_fraud else []) + [
        "TX_DURING_WEEKEND", "TX_DURING_NIGHT"]
    st.dataframe(filtered_df[explorer_cols].head(PREVIEW_ROWS), use_container_width=True, hide_index=True,
                 column_config={"TRANSACTION_ID": st.column_config.NumberColumn("TRANSACTION_ID", format="%d")})
    st.download_button("📥 Download Preprocessed CSV", data=filtered_csv(file_key, file_bytes, d0, d1, a0, a1),
                       file_name="preprocessed_transactions.csv", mime="text/csv")

# TAB 3: FEATURE ENGINEERING MATRIX
with tab3:
    st.subheader("Engineered Feature Schema")
    st.markdown("Features generated automatically for downstream ML modeling. Terminal counts include only "
                "**earlier** transactions, never the current one.")
    st.dataframe(filtered_df.drop(columns=["TX_PERIOD"]).head(100), use_container_width=True)

# TAB 4: FRAUD TRANSACTIONS (only when the CSV has TX_FRAUD)
if has_fraud:
    with tabs[3]:
        st.subheader(f"{nf:,} fraudulent transactions in the current filters")
        look = st.text_input("Look up any Transaction ID", placeholder="e.g. 1546297",
                             help="Checks whether a transaction in the current filters is labeled fraud or legitimate.")
        if look.strip():
            if look.strip().isdigit():
                hit = filtered_df[filtered_df["TRANSACTION_ID"] == int(look.strip())]
                if hit.empty:
                    st.warning("That ID is not in the current filtered data (it may be filtered out or removed during cleaning).")
                else:
                    r = hit.iloc[0]
                    (st.error if r["TX_FRAUD"] == 1 else st.success)(
                        f"Transaction {r['TRANSACTION_ID']} is labeled {'FRAUD' if r['TX_FRAUD'] == 1 else 'legitimate'}: "
                        f"${r['TX_AMOUNT']:,.2f} on {r['TX_DATETIME']:%Y-%m-%d %H:%M}, customer {r['CUSTOMER_ID']}, terminal {r['TERMINAL_ID']}.")
            else:
                st.warning("Transaction IDs are whole numbers.")

        if nf == 0:
            st.info("No fraud transactions in the current filters.")
        else:
            show = fraud_df[["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]]
            st.caption(f"Showing the first {min(PREVIEW_ROWS, nf):,} of {nf:,} fraud rows. The download contains all of them.")
            st.dataframe(show.head(PREVIEW_ROWS), use_container_width=True, hide_index=True,
                         column_config={"TRANSACTION_ID": st.column_config.NumberColumn("TRANSACTION_ID", format="%d")})
            st.download_button("📥 Download fraud transaction IDs (CSV)", data=fraud_csv(file_key, file_bytes, d0, d1, a0, a1),
                               file_name="fraud_transactions.csv", mime="text/csv")

            c1, c2 = st.columns(2)
            top = fraud_df.groupby("TERMINAL_ID").size().nlargest(10).reset_index(name="FRAUD_COUNT")
            fig_t = px.bar(top, x="TERMINAL_ID", y="FRAUD_COUNT", title="<b>Top 10 terminals by fraud count</b>",
                           labels={"TERMINAL_ID": "Terminal", "FRAUD_COUNT": "Fraud transactions"}, template="plotly_dark")
            fig_t.update_traces(marker_color="#EF4444")
            fig_t.update_xaxes(type="category")
            c1.plotly_chart(fig_t, use_container_width=True)
            by_hour = fraud_df.groupby("TX_HOUR").size().reindex(range(24), fill_value=0).reset_index(name="FRAUD_COUNT")
            fig_h = px.bar(by_hour, x="TX_HOUR", y="FRAUD_COUNT", title="<b>Fraud transactions by hour of day</b>",
                           labels={"TX_HOUR": "Hour of Day (0-23)", "FRAUD_COUNT": "Fraud transactions"}, template="plotly_dark")
            fig_h.update_traces(marker_color="#F59E0B")
            c2.plotly_chart(fig_h, use_container_width=True)