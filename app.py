# app.py — Stock Dashboard (UI nâng cấp + ChatGPT)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import plotly.graph_objects as go
import os
from openai import OpenAI

# Prophet (tùy chọn)
HAS_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    HAS_PROPHET = False

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== THEME & CSS ===================
THEME_CSS = """
<style>
.stApp { background: #0f1220; color: #e8eef9; }
h1, h2, h3 { color: #ffd166; }
hr { border: 0; height: 1px; background: #22263a; }
.card {
  background: #151936; border: 1px solid #23284b; border-radius: 16px;
  padding: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.small { font-size: 12px; color: #9aa4bf; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.title("📊 Stock Price Dashboard & Forecast")

# =================== SIDEBAR ===================
st.sidebar.header("🧭 Điều hướng")
page = st.sidebar.radio(
    "Chọn trang",
    ["📊 Dashboard", "🔮 Dự báo", "🤖 ChatGPT"],
    index=0
)

st.sidebar.header("📂 Dữ liệu")
uploaded = st.sidebar.file_uploader("Tải CSV (all_stocks_5yr.csv)", type=["csv"])

date_filter_on = st.sidebar.checkbox("Lọc theo khoảng ngày", value=False)
start_date = st.sidebar.date_input("Từ ngày", value=date(2016, 1, 1))
end_date   = st.sidebar.date_input("Đến ngày", value=date(2018, 2, 7))

st.sidebar.markdown("---")

# =================== LOAD DATA ===================
@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    if "name" not in df.columns and "Name" in df.columns:
        df["name"] = df["Name"]
    df["date"] = pd.to_datetime(df["date"])
    return df

if uploaded is None and page != "🤖 ChatGPT":
    st.info("⬆️ Hãy tải file CSV để bắt đầu.")
    st.stop()

if uploaded is not None:
    df = load_df(uploaded)
    if "name" not in df.columns:
        st.error("Không tìm thấy cột 'name' (mã cổ phiếu). Hãy kiểm tra CSV.")
        st.stop()

    tickers = sorted(df["name"].unique().tolist())
    ticker = st.sidebar.selectbox("Chọn mã cổ phiếu", options=tickers, index=0)

    df_t = df[df["name"] == ticker].copy()
    if date_filter_on:
        df_t = df_t[(df_t["date"] >= pd.to_datetime(start_date)) & (df_t["date"] <= pd.to_datetime(end_date))]

    if df_t.empty:
        st.warning("Không có dữ liệu sau khi lọc. Hãy đổi mã/tùy chọn ngày.")
        st.stop()

# =================== METRICS (cards) ===================
def render_metrics(df_t):
    last_close = float(df_t["close"].iloc[-1])
    prev_close = float(df_t["close"].iloc[-2]) if len(df_t) > 1 else last_close
    chg_pct_1d = (last_close/prev_close - 1) * 100 if prev_close else 0.0

    if len(df_t) > 7:
        last_7 = float(df_t["close"].iloc[-7])
        chg_pct_7d = (last_close/last_7 - 1) * 100
    else:
        chg_pct_7d = np.nan

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="card"><h3>Giá hiện tại</h3>
        <div style="font-size:28px;font-weight:700">${last_close:,.2f}</div>
        <div class="small">Close gần nhất</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><h3>Thay đổi 1 ngày</h3>
        <div style="font-size:28px;font-weight:700">{chg_pct_1d:+.2f}%</div>
        <div class="small">So với phiên trước</div></div>""", unsafe_allow_html=True)
    with c3:
        val7 = "-" if np.isnan(chg_pct_7d) else f"{chg_pct_7d:+.2f}%"
        st.markdown(f"""<div class="card"><h3>Thay đổi 7 ngày</h3>
        <div style="font-size:28px;font-weight:700">{val7}</div>
        <div class="small">So với 7 phiên trước</div></div>""", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

# =================== PLOTS ===================
def plot_candlestick(_df: pd.DataFrame, ticker: str):
    fig = go.Figure(data=[go.Candlestick(
        x=_df["date"],
        open=_df["open"],
        high=_df["high"],
        low=_df["low"],
        close=_df["close"],
        name="OHLC"
    )])
    fig.update_layout(
        title=f"Candlestick — {ticker}",
        xaxis_title="Ngày",
        yaxis_title="Giá (USD)",
        margin=dict(l=10,r=10,t=40,b=10),
        height=480
    )
    return fig

def plot_volume(_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=_df["date"], y=_df["volume"], name="Volume"))
    fig.update_layout(
        title="Khối lượng giao dịch",
        xaxis_title="Ngày",
        yaxis_title="Số lượng",
        margin=dict(l=10,r=10,t=40,b=10),
        height=240
    )
    return fig

# =================== CHATGPT TAB ===================
def chat_tab():
    st.header("🤖 ChatGPT trợ lý phân tích")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Thiếu OPENAI_API_KEY. Hãy cấu hình biến môi trường trên server.")
        return

    client = OpenAI(api_key=api_key)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": "Bạn là chuyên gia phân tích tài chính, trả lời ngắn gọn, rõ ràng."}
        ]

    # hiển thị lịch sử
    for m in st.session_state.chat_messages:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        elif m["role"] in ("assistant", "system"):
            with st.chat_message("assistant"):
                st.markdown(m["content"])

    prompt = st.chat_input("Nhập câu hỏi (ví dụ: 'Phân tích cổ phiếu AAPL')")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.chat_messages,
                    temperature=0.3
                )
                answer = resp.choices[0].message.content
                st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# =================== PAGE ROUTING ===================
if page == "📊 Dashboard" and uploaded is not None:
    render_metrics(df_t)
    tab1, tab2 = st.tabs(["📈 Giá & Khối lượng", "📋 Thống kê mô tả"])
    with tab1:
        st.plotly_chart(plot_candlestick(df_t, ticker), use_container_width=True)
        st.plotly_chart(plot_volume(df_t), use_container_width=True)
    with tab2:
        st.dataframe(df_t[["open","high","low","close","volume"]].describe().round(3))

elif page == "🔮 Dự báo" and uploaded is not None:
    if not HAS_PROPHET:
        st.error("Prophet chưa được cài. Chạy: pip install prophet")
        st.stop()

    st.subheader(f"🔮 Dự báo với Prophet — {ticker}")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        horizon = st.slider("Số ngày dự báo", 30, 730, 365, step=15)
    with colB:
        weekly = st.checkbox("Weekly seasonality", value=True)
    with colC:
        show_ci = st.checkbox("Hiển thị khoảng tin cậy", value=True)

    data_p = df_t[["date","close"]].rename(columns={"date":"ds","close":"y"}).reset_index(drop=True)
    m = Prophet(daily_seasonality=False, weekly_seasonality=weekly, yearly_seasonality=True)
    m.fit(data_p)

    future = m.make_future_dataframe(periods=horizon, freq="B")
    fc = m.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_p["ds"], y=data_p["y"], mode="lines", name="Giá thực tế"))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Giá dự báo"))

    if show_ci:
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 name="Khoảng tin cậy", hoverinfo="skip"))

    fig.update_layout(
        title=f"Dự báo {horizon} ngày tiếp theo",
        xaxis_title="Ngày", yaxis_title="Giá đóng cửa (USD)",
        margin=dict(l=10,r=10,t=40,b=10), height=520
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📑 Xem bảng dự báo (cuối)"):
        st.dataframe(fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(30))

elif page == "🤖 ChatGPT":
    chat_tab()
