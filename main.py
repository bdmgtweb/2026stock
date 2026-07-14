import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="Interactive Stock Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS 디자인
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 32%),
                radial-gradient(circle at top right, rgba(168, 85, 247, 0.10), transparent 28%);
        }

        .main-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            letter-spacing: -0.04em;
        }

        .sub-title {
            color: #8b95a7;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .stock-card {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 18px;
            padding: 18px 20px;
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(8px);
            margin-bottom: 12px;
        }

        .signal-positive {
            padding: 9px 13px;
            border-radius: 10px;
            background-color: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.30);
        }

        .signal-negative {
            padding: 9px 13px;
            border-radius: 10px;
            background-color: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.30);
        }

        .signal-neutral {
            padding: 9px 13px;
            border-radius: 10px;
            background-color: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.30);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.035);
        }

        div[data-testid="stMetricLabel"] {
            color: #8b95a7;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 티커 목록
# =========================================================
TICKER_PRESETS = {
    "직접 입력": "",
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "현대차": "005380.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "셀트리온": "068270.KS",
    "에코프로비엠": "247540.KQ",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Palantir": "PLTR",
    "Broadcom": "AVGO",
    "S&P 500 ETF": "SPY",
    "NASDAQ 100 ETF": "QQQ",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "5년": "5y",
    "10년": "10y",
    "전체": "max",
}

INTERVAL_OPTIONS = {
    "일봉": "1d",
    "주봉": "1wk",
    "월봉": "1mo",
}


# =========================================================
# 데이터 관련 함수
# =========================================================
def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """yfinance 결과가 MultiIndex일 경우 일반 컬럼으로 변환합니다."""
    if isinstance(data.columns, pd.MultiIndex):
        level_zero = data.columns.get_level_values(0)

        expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if len(expected.intersection(set(level_zero))) > 0:
            data.columns = level_zero
        else:
            data.columns = data.columns.get_level_values(-1)

    return data


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(
    ticker: str,
    period: str,
    interval: str,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Yahoo Finance에서 주가 데이터를 불러옵니다."""
    ticker = ticker.strip().upper()

    try:
        if start_date is not None and end_date is not None:
            # yfinance의 end는 일반적으로 해당 날짜를 포함하지 않으므로 하루를 더합니다.
            inclusive_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            data = yf.download(
                ticker,
                start=pd.Timestamp(start_date),
                end=inclusive_end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        else:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

        if data is None or data.empty:
            return pd.DataFrame()

        data = flatten_yfinance_columns(data.copy())
        data.index = pd.to_datetime(data.index)

        # 시간대 정보가 있을 경우 제거하여 Plotly와 CSV에서 안정적으로 처리
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        for column in required_columns:
            if column not in data.columns:
                return pd.DataFrame()

        data = data.loc[:, ~data.columns.duplicated()].copy()
        data = data.dropna(subset=["Close"])

        numeric_columns = [
            column
            for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if column in data.columns
        ]

        data[numeric_columns] = data[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_info(ticker: str) -> dict:
    """종목의 기본 정보를 가져옵니다. 실패하면 빈 딕셔너리를 반환합니다."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not isinstance(info, dict):
            return {}

        return info

    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_comparison_data(
    tickers: tuple,
    period: str,
) -> pd.DataFrame:
    """여러 종목의 종가를 불러와 비교 분석에 사용합니다."""
    valid_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]

    if len(valid_tickers) == 0:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=valid_tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if len(valid_tickers) == 1:
            data = flatten_yfinance_columns(data)

            if "Close" not in data.columns:
                return pd.DataFrame()

            close_data = data[["Close"]].copy()
            close_data.columns = [valid_tickers[0]]

        else:
            if not isinstance(data.columns, pd.MultiIndex):
                return pd.DataFrame()

            if "Close" in data.columns.get_level_values(0):
                close_data = data["Close"].copy()
            elif "Close" in data.columns.get_level_values(1):
                close_data = data.xs("Close", axis=1, level=1).copy()
            else:
                return pd.DataFrame()

        close_data.index = pd.to_datetime(close_data.index)

        if getattr(close_data.index, "tz", None) is not None:
            close_data.index = close_data.index.tz_localize(None)

        close_data = close_data.apply(pd.to_numeric, errors="coerce")
        close_data = close_data.dropna(how="all")

        return close_data

    except Exception:
        return pd.DataFrame()


# =========================================================
# 기술적 지표 함수
# =========================================================
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"].fillna(0)).cumprod() - 1

    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA60"] = df["Close"].rolling(window=60).mean()
    df["SMA120"] = df["Close"].rolling(window=120).mean()

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD Histogram"] = df["MACD"] - df["MACD Signal"]

    rolling_std = df["Close"].rolling(window=20).std()
    df["Bollinger Upper"] = df["SMA20"] + 2 * rolling_std
    df["Bollinger Lower"] = df["SMA20"] - 2 * rolling_std

    df["RSI"] = calculate_rsi(df["Close"])

    df["Previous Close"] = df["Close"].shift(1)
    df["True Range"] = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Previous Close"]).abs(),
            (df["Low"] - df["Previous Close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = df["True Range"].rolling(window=14).mean()

    rolling_peak = df["Close"].cummax()
    df["Drawdown"] = df["Close"] / rolling_peak - 1

    return df


def safe_number(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def format_money(value: float, currency: str) -> str:
    if pd.isna(value):
        return "-"

    symbol = {
        "KRW": "₩",
        "USD": "$",
        "JPY": "¥",
        "EUR": "€",
        "GBP": "£",
    }.get(currency, "")

    if currency in ["KRW", "JPY"]:
        return f"{symbol}{value:,.0f}"

    return f"{symbol}{value:,.2f}"


def format_large_number(value) -> str:
    value = safe_number(value)

    if pd.isna(value):
        return "-"

    absolute = abs(value)

    if absolute >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:,.2f}T"
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:,.2f}K"

    return f"{value:,.0f}"


def annualization_factor(interval: str) -> int:
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    return 252


def create_signal_summary(df: pd.DataFrame) -> list:
    latest = df.iloc[-1]
    signals = []

    close = latest["Close"]
    sma20 = latest.get("SMA20", np.nan)
    sma60 = latest.get("SMA60", np.nan)
    rsi = latest.get("RSI", np.nan)
    macd = latest.get("MACD", np.nan)
    signal = latest.get("MACD Signal", np.nan)

    if not pd.isna(sma20):
        if close > sma20:
            signals.append(("긍정", "현재 가격이 20기간 이동평균선 위에 있습니다."))
        else:
            signals.append(("주의", "현재 가격이 20기간 이동평균선 아래에 있습니다."))

    if not pd.isna(sma20) and not pd.isna(sma60):
        if sma20 > sma60:
            signals.append(("긍정", "단기 이동평균선이 중기 이동평균선 위에 있습니다."))
        else:
            signals.append(("주의", "단기 이동평균선이 중기 이동평균선 아래에 있습니다."))

    if not pd.isna(rsi):
        if rsi >= 70:
            signals.append(("과열", f"RSI가 {rsi:.1f}로 과매수 구간에 있습니다."))
        elif rsi <= 30:
            signals.append(("침체", f"RSI가 {rsi:.1f}로 과매도 구간에 있습니다."))
        else:
            signals.append(("중립", f"RSI가 {rsi:.1f}로 중립 구간에 있습니다."))

    if not pd.isna(macd) and not pd.isna(signal):
        if macd > signal:
            signals.append(("긍정", "MACD가 시그널선 위에 있습니다."))
        else:
            signals.append(("주의", "MACD가 시그널선 아래에 있습니다."))

    return signals


def make_price_chart(
    df: pd.DataFrame,
    ticker: str,
    chart_type: str,
    show_volume: bool,
    show_sma20: bool,
    show_sma60: bool,
    show_sma120: bool,
    show_bollinger: bool,
) -> go.Figure:
    rows = 2 if show_volume else 1
    row_heights = [0.76, 0.24] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    if chart_type == "캔들 차트":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="종가",
                mode="lines",
                line=dict(width=2.2, color="#60a5fa"),
                hovertemplate="%{x|%Y-%m-%d}<br>종가: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if show_sma20:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA20"],
                name="SMA 20",
                mode="lines",
                line=dict(width=1.5, color="#f59e0b"),
            ),
            row=1,
            col=1,
        )

    if show_sma60:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA60"],
                name="SMA 60",
                mode="lines",
                line=dict(width=1.5, color="#a855f7"),
            ),
            row=1,
            col=1,
        )

    if show_sma120:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA120"],
                name="SMA 120",
                mode="lines",
                line=dict(width=1.5, color="#14b8a6"),
            ),
            row=1,
            col=1,
        )

    if show_bollinger:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Bollinger Upper"],
                name="볼린저 상단",
                mode="lines",
                line=dict(width=1, dash="dot", color="rgba(148,163,184,0.7)"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Bollinger Lower"],
                name="볼린저 하단",
                mode="lines",
                line=dict(width=1, dash="dot", color="rgba(148,163,184,0.7)"),
                fill="tonexty",
                fillcolor="rgba(96,165,250,0.08)",
            ),
            row=1,
            col=1,
        )

    if show_volume:
        volume_colors = np.where(
            df["Close"] >= df["Open"],
            "rgba(34,197,94,0.65)",
            "rgba(239,68,68,0.65)",
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="거래량",
                marker_color=volume_colors,
                hovertemplate="%{x|%Y-%m-%d}<br>거래량: %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="거래량", row=2, col=1)

    fig.update_yaxes(title_text="가격", row=1, col=1)

    fig.update_layout(
        title=dict(
            text=f"{ticker} 가격 차트",
            x=0.01,
            xanchor="left",
        ),
        height=700 if show_volume else 570,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
            bgcolor="rgba(30,41,59,0.85)",
            activecolor="#2563eb",
            font=dict(color="white"),
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
    )

    return fig


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.markdown("## 📊 분석 설정")

    selected_name = st.selectbox(
        "종목 빠른 선택",
        options=list(TICKER_PRESETS.keys()),
        index=list(TICKER_PRESETS.keys()).index("NVIDIA"),
    )

    default_ticker = TICKER_PRESETS[selected_name]

    ticker = st.text_input(
        "Yahoo Finance 티커",
        value=default_ticker,
        placeholder="예: AAPL, 005930.KS",
        help=(
            "코스피는 종목코드 뒤에 .KS, 코스닥은 .KQ를 붙입니다. "
            "예: 삼성전자 005930.KS"
        ),
    ).strip().upper()

    date_mode = st.radio(
        "조회 기간 설정",
        ["간편 기간", "날짜 직접 선택"],
        horizontal=True,
    )

    if date_mode == "간편 기간":
        period_label = st.selectbox(
            "조회 기간",
            list(PERIOD_OPTIONS.keys()),
            index=3,
        )
        period = PERIOD_OPTIONS[period_label]
        start_date = None
        end_date = None
    else:
        default_start = date.today() - timedelta(days=365)
        default_end = date.today()

        start_date = st.date_input(
            "시작일",
            value=default_start,
            max_value=default_end,
        )

        end_date = st.date_input(
            "종료일",
            value=default_end,
            min_value=start_date,
            max_value=default_end,
        )

        period = "1y"

    interval_label = st.selectbox(
        "데이터 간격",
        list(INTERVAL_OPTIONS.keys()),
        index=0,
    )
    interval = INTERVAL_OPTIONS[interval_label]

    chart_type = st.radio(
        "차트 종류",
        ["캔들 차트", "라인 차트"],
        horizontal=True,
    )

    st.markdown("---")
    st.markdown("### 기술적 지표")

    show_volume = st.checkbox("거래량", value=True)
    show_sma20 = st.checkbox("20기간 이동평균", value=True)
    show_sma60 = st.checkbox("60기간 이동평균", value=True)
    show_sma120 = st.checkbox("120기간 이동평균", value=False)
    show_bollinger = st.checkbox("볼린저 밴드", value=True)

    st.markdown("---")

    refresh = st.button(
        "🔄 최신 데이터 새로고침",
        use_container_width=True,
    )

    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "한국 주식 예시: 삼성전자 005930.KS · "
        "SK하이닉스 000660.KS · 에코프로비엠 247540.KQ"
    )


# =========================================================
# 제목
# =========================================================
st.markdown(
    '<div class="main-title">📈 Interactive Stock Lab</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "Yahoo Finance 데이터와 Plotly로 분석하는 인터랙티브 주식 대시보드"
    "</div>",
    unsafe_allow_html=True,
)


if not ticker:
    st.info("왼쪽 사이드바에서 분석할 종목 티커를 입력해 주세요.")
    st.stop()


# =========================================================
# 데이터 로드
# =========================================================
with st.spinner(f"{ticker} 데이터를 불러오는 중입니다..."):
    raw_data = load_stock_data(
        ticker=ticker,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )

if raw_data.empty:
    st.error(
        "주가 데이터를 불러오지 못했습니다.\n\n"
        "티커가 올바른지 확인해 주세요. "
        "코스피 종목은 `.KS`, 코스닥 종목은 `.KQ`를 붙여야 합니다.\n\n"
        "예: 삼성전자 `005930.KS`, 에코프로비엠 `247540.KQ`"
    )
    st.stop()

df = add_indicators(raw_data)

with st.spinner("기업 정보를 확인하는 중입니다..."):
    company_info = load_company_info(ticker)


# =========================================================
# 기본 정보
# =========================================================
company_name = (
    company_info.get("longName")
    or company_info.get("shortName")
    or ticker
)

currency = company_info.get("currency", "")
exchange = (
    company_info.get("exchange")
    or company_info.get("fullExchangeName")
    or "-"
)

sector = company_info.get("sector", "-")
industry = company_info.get("industry", "-")

st.markdown(
    f"""
    <div class="stock-card">
        <h2 style="margin:0 0 6px 0;">{company_name}</h2>
        <div style="color:#8b95a7;">
            {ticker} · {exchange} · {sector} · {industry}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 핵심 지표 계산
# =========================================================
latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) >= 2 else latest

latest_close = safe_number(latest["Close"])
previous_close = safe_number(previous["Close"])

price_change = latest_close - previous_close
price_change_pct = (
    price_change / previous_close * 100
    if previous_close != 0 and not pd.isna(previous_close)
    else 0
)

period_high = safe_number(df["High"].max())
period_low = safe_number(df["Low"].min())

returns = df["Daily Return"].dropna()
factor = annualization_factor(interval)

annual_volatility = (
    returns.std() * np.sqrt(factor) * 100
    if len(returns) > 1
    else np.nan
)

total_return = (
    (latest_close / safe_number(df["Close"].iloc[0]) - 1) * 100
    if safe_number(df["Close"].iloc[0]) != 0
    else np.nan
)

max_drawdown = safe_number(df["Drawdown"].min()) * 100
latest_volume = safe_number(latest["Volume"])
average_volume = safe_number(df["Volume"].tail(20).mean())


# =========================================================
# 핵심 지표 표시
# =========================================================
metric_columns = st.columns(6)

metric_columns[0].metric(
    "현재 가격",
    format_money(latest_close, currency),
    f"{price_change_pct:+.2f}%",
)

metric_columns[1].metric(
    "기간 수익률",
    f"{total_return:+.2f}%",
)

metric_columns[2].metric(
    "기간 최고가",
    format_money(period_high, currency),
)

metric_columns[3].metric(
    "기간 최저가",
    format_money(period_low, currency),
)

metric_columns[4].metric(
    "연환산 변동성",
    f"{annual_volatility:.2f}%"
    if not pd.isna(annual_volatility)
    else "-",
)

metric_columns[5].metric(
    "최대 낙폭",
    f"{max_drawdown:.2f}%"
    if not pd.isna(max_drawdown)
    else "-",
)


# =========================================================
# 탭 구성
# =========================================================
tabs = st.tabs(
    [
        "📈 가격 차트",
        "🧭 기술적 분석",
        "⚖️ 종목 비교",
        "📊 위험·수익 분석",
        "🏢 기업 정보",
        "🗂️ 원본 데이터",
    ]
)


# =========================================================
# 탭 1: 가격 차트
# =========================================================
with tabs[0]:
    price_chart = make_price_chart(
        df=df,
        ticker=ticker,
        chart_type=chart_type,
        show_volume=show_volume,
        show_sma20=show_sma20,
        show_sma60=show_sma60,
        show_sma120=show_sma120,
        show_bollinger=show_bollinger,
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    detail_columns = st.columns(4)

    detail_columns[0].metric(
        "시가",
        format_money(safe_number(latest["Open"]), currency),
    )

    detail_columns[1].metric(
        "고가",
        format_money(safe_number(latest["High"]), currency),
    )

    detail_columns[2].metric(
        "저가",
        format_money(safe_number(latest["Low"]), currency),
    )

    detail_columns[3].metric(
        "거래량",
        format_large_number(latest_volume),
        (
            f"{(latest_volume / average_volume - 1) * 100:+.1f}% vs 20기간 평균"
            if average_volume not in [0, np.nan] and not pd.isna(average_volume)
            else None
        ),
    )


# =========================================================
# 탭 2: 기술적 분석
# =========================================================
with tabs[1]:
    signal_col, indicator_col = st.columns([0.36, 0.64])

    with signal_col:
        st.markdown("### 자동 지표 해석")
        st.caption("아래 내용은 선택한 기술지표를 단순 규칙으로 정리한 것입니다.")

        for signal_type, message in create_signal_summary(df):
            if signal_type == "긍정":
                css_class = "signal-positive"
                icon = "🟢"
            elif signal_type in ["주의", "과열"]:
                css_class = "signal-negative"
                icon = "🔴"
            else:
                css_class = "signal-neutral"
                icon = "🔵"

            st.markdown(
                f"""
                <div class="{css_class}" style="margin-bottom:10px;">
                    <strong>{icon} {signal_type}</strong><br>
                    <span style="font-size:0.92rem;">{message}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 최신 지표")

        indicator_metrics = st.columns(2)

        indicator_metrics[0].metric(
            "RSI(14)",
            f"{safe_number(latest['RSI']):.2f}",
        )

        indicator_metrics[1].metric(
            "ATR(14)",
            format_money(safe_number(latest["ATR14"]), currency),
        )

        indicator_metrics[0].metric(
            "MACD",
            f"{safe_number(latest['MACD']):.3f}",
        )

        indicator_metrics[1].metric(
            "MACD Signal",
            f"{safe_number(latest['MACD Signal']):.3f}",
        )

    with indicator_col:
        rsi_fig = go.Figure()

        rsi_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI",
                mode="lines",
                line=dict(color="#60a5fa", width=2),
            )
        )

        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="과매수 70",
        )

        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text="과매도 30",
        )

        rsi_fig.add_hrect(
            y0=30,
            y1=70,
            fillcolor="rgba(96,165,250,0.05)",
            line_width=0,
        )

        rsi_fig.update_layout(
            title="RSI 상대강도지수",
            template="plotly_dark",
            height=330,
            margin=dict(l=20, r=20, t=55, b=20),
            yaxis=dict(range=[0, 100]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )

        st.plotly_chart(
            rsi_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    macd_fig = make_subplots(specs=[[{"secondary_y": False}]])

    macd_colors = np.where(
        df["MACD Histogram"] >= 0,
        "rgba(34,197,94,0.75)",
        "rgba(239,68,68,0.75)",
    )

    macd_fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD Histogram"],
            name="히스토그램",
            marker_color=macd_colors,
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(color="#60a5fa", width=2),
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD Signal"],
            name="Signal",
            line=dict(color="#f59e0b", width=1.8),
        )
    )

    macd_fig.add_hline(
        y=0,
        line_color="rgba(148,163,184,0.5)",
        line_width=1,
    )

    macd_fig.update_layout(
        title="MACD 분석",
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h"),
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )


# =========================================================
# 탭 3: 종목 비교
# =========================================================
with tabs[2]:
    st.markdown("### 여러 종목의 상대 수익률 비교")

    default_comparison = f"{ticker}, SPY, QQQ"

    comparison_input = st.text_input(
        "비교할 티커를 쉼표로 구분해 입력하세요",
        value=default_comparison,
        help="예: AAPL, MSFT, NVDA 또는 005930.KS, 000660.KS",
    )

    comparison_period_label = st.selectbox(
        "비교 기간",
        ["3개월", "6개월", "1년", "2년", "5년"],
        index=2,
    )

    comparison_period = PERIOD_OPTIONS[comparison_period_label]

    comparison_tickers = tuple(
        dict.fromkeys(
            ticker_item.strip().upper()
            for ticker_item in comparison_input.split(",")
            if ticker_item.strip()
        )
    )

    if len(comparison_tickers) > 8:
        st.warning("차트 가독성을 위해 비교 종목은 최대 8개까지만 사용합니다.")
        comparison_tickers = comparison_tickers[:8]

    comparison_data = load_comparison_data(
        comparison_tickers,
        comparison_period,
    )

    if comparison_data.empty:
        st.warning("비교 데이터를 불러오지 못했습니다. 티커를 확인해 주세요.")

    else:
        normalized_data = comparison_data.copy()

        for column in normalized_data.columns:
            first_valid = normalized_data[column].dropna()

            if not first_valid.empty and first_valid.iloc[0] != 0:
                normalized_data[column] = (
                    normalized_data[column] / first_valid.iloc[0] * 100
                )

        normalized_long = (
            normalized_data
            .reset_index()
            .melt(
                id_vars=normalized_data.index.name or "Date",
                var_name="Ticker",
                value_name="Normalized",
            )
        )

        date_column = normalized_long.columns[0]

        comparison_fig = px.line(
            normalized_long,
            x=date_column,
            y="Normalized",
            color="Ticker",
            title="기준일을 100으로 환산한 상대 성과",
        )

        comparison_fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="rgba(148,163,184,0.5)",
        )

        comparison_fig.update_layout(
            template="plotly_dark",
            height=550,
            margin=dict(l=20, r=20, t=65, b=20),
            hovermode="x unified",
            xaxis_title="날짜",
            yaxis_title="정규화 가격",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h"),
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

        comparison_returns = comparison_data.pct_change().dropna(how="all")
        comparison_summary = []

        for column in comparison_data.columns:
            price_series = comparison_data[column].dropna()
            return_series = comparison_returns[column].dropna()

            if len(price_series) < 2:
                continue

            result = {
                "종목": column,
                "기간 수익률(%)": (
                    price_series.iloc[-1] / price_series.iloc[0] - 1
                ) * 100,
                "연환산 변동성(%)": (
                    return_series.std() * np.sqrt(252) * 100
                    if len(return_series) > 1
                    else np.nan
                ),
                "최고가": price_series.max(),
                "최저가": price_series.min(),
            }

            comparison_summary.append(result)

        summary_df = pd.DataFrame(comparison_summary)

        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                "기간 수익률(%)",
                ascending=False,
            )

            st.dataframe(
                summary_df.style.format(
                    {
                        "기간 수익률(%)": "{:+.2f}",
                        "연환산 변동성(%)": "{:.2f}",
                        "최고가": "{:,.2f}",
                        "최저가": "{:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        if len(comparison_returns.columns) >= 2:
            correlation = comparison_returns.corr()

            heatmap = go.Figure(
                data=go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.index,
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    reversescale=True,
                    text=np.round(correlation.values, 2),
                    texttemplate="%{text}",
                    hovertemplate=(
                        "%{x} / %{y}<br>상관계수: %{z:.2f}<extra></extra>"
                    ),
                )
            )

            heatmap.update_layout(
                title="일간 수익률 상관관계",
                template="plotly_dark",
                height=480,
                margin=dict(l=20, r=20, t=60, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True,
                config={"displaylogo": False},
            )


# =========================================================
# 탭 4: 위험·수익 분석
# =========================================================
with tabs[3]:
    analysis_left, analysis_right = st.columns(2)

    with analysis_left:
        return_fig = go.Figure()

        return_fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name="일간 수익률",
                marker_color="#60a5fa",
                opacity=0.82,
                hovertemplate="수익률: %{x:.2f}%<br>빈도: %{y}<extra></extra>",
            )
        )

        return_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="white",
        )

        return_fig.update_layout(
            title="수익률 분포",
            template="plotly_dark",
            height=420,
            xaxis_title="수익률(%)",
            yaxis_title="빈도",
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.04,
        )

        st.plotly_chart(
            return_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with analysis_right:
        cumulative_fig = go.Figure()

        cumulative_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cumulative Return"] * 100,
                name="누적 수익률",
                mode="lines",
                fill="tozeroy",
                line=dict(color="#22c55e", width=2),
                fillcolor="rgba(34,197,94,0.12)",
            )
        )

        cumulative_fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="rgba(148,163,184,0.7)",
        )

        cumulative_fig.update_layout(
            title="누적 수익률",
            template="plotly_dark",
            height=420,
            xaxis_title="날짜",
            yaxis_title="누적 수익률(%)",
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            cumulative_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    drawdown_fig = go.Figure()

    drawdown_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown"] * 100,
            name="낙폭",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ef4444", width=1.8),
            fillcolor="rgba(239,68,68,0.17)",
            hovertemplate="%{x|%Y-%m-%d}<br>낙폭: %{y:.2f}%<extra></extra>",
        )
    )

    drawdown_fig.update_layout(
        title="고점 대비 낙폭 분석",
        template="plotly_dark",
        height=400,
        xaxis_title="날짜",
        yaxis_title="낙폭(%)",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        drawdown_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    if len(returns) > 1:
        positive_ratio = (returns > 0).mean() * 100
        best_period = returns.max() * 100
        worst_period = returns.min() * 100
        average_return = returns.mean() * 100

        downside_returns = returns[returns < 0]

        downside_deviation = (
            downside_returns.std() * np.sqrt(factor)
            if len(downside_returns) > 1
            else np.nan
        )

        annual_return = returns.mean() * factor

        sortino_ratio = (
            annual_return / downside_deviation
            if not pd.isna(downside_deviation) and downside_deviation != 0
            else np.nan
        )

        risk_metrics = st.columns(5)

        risk_metrics[0].metric(
            "상승 기간 비율",
            f"{positive_ratio:.1f}%",
        )

        risk_metrics[1].metric(
            "평균 기간 수익률",
            f"{average_return:+.3f}%",
        )

        risk_metrics[2].metric(
            "최고 기간 수익률",
            f"{best_period:+.2f}%",
        )

        risk_metrics[3].metric(
            "최저 기간 수익률",
            f"{worst_period:+.2f}%",
        )

        risk_metrics[4].metric(
            "Sortino Ratio",
            f"{sortino_ratio:.2f}"
            if not pd.isna(sortino_ratio)
            else "-",
        )


# =========================================================
# 탭 5: 기업 정보
# =========================================================
with tabs[4]:
    if not company_info:
        st.info(
            "Yahoo Finance에서 기업 상세 정보를 가져오지 못했습니다. "
            "가격 데이터 분석 기능은 계속 사용할 수 있습니다."
        )

    else:
        info_columns = st.columns(4)

        market_cap = company_info.get("marketCap")
        enterprise_value = company_info.get("enterpriseValue")
        trailing_pe = company_info.get("trailingPE")
        forward_pe = company_info.get("forwardPE")
        price_to_book = company_info.get("priceToBook")
        dividend_yield = company_info.get("dividendYield")
        beta = company_info.get("beta")
        fifty_two_week_high = company_info.get("fiftyTwoWeekHigh")
        fifty_two_week_low = company_info.get("fiftyTwoWeekLow")
        target_mean_price = company_info.get("targetMeanPrice")

        info_columns[0].metric(
            "시가총액",
            format_large_number(market_cap),
        )

        info_columns[1].metric(
            "기업가치",
            format_large_number(enterprise_value),
        )

        info_columns[2].metric(
            "과거 PER",
            f"{safe_number(trailing_pe):.2f}"
            if not pd.isna(safe_number(trailing_pe))
            else "-",
        )

        info_columns[3].metric(
            "예상 PER",
            f"{safe_number(forward_pe):.2f}"
            if not pd.isna(safe_number(forward_pe))
            else "-",
        )

        second_info_columns = st.columns(4)

        second_info_columns[0].metric(
            "PBR",
            f"{safe_number(price_to_book):.2f}"
            if not pd.isna(safe_number(price_to_book))
            else "-",
        )

        second_info_columns[1].metric(
            "배당수익률",
            (
                f"{safe_number(dividend_yield) * 100:.2f}%"
                if not pd.isna(safe_number(dividend_yield))
                else "-"
            ),
        )

        second_info_columns[2].metric(
            "베타",
            f"{safe_number(beta):.2f}"
            if not pd.isna(safe_number(beta))
            else "-",
        )

        second_info_columns[3].metric(
            "목표주가 평균",
            (
                format_money(
                    safe_number(target_mean_price),
                    currency,
                )
                if not pd.isna(safe_number(target_mean_price))
                else "-"
            ),
        )

        price_range_columns = st.columns(2)

        price_range_columns[0].metric(
            "52주 최고가",
            (
                format_money(
                    safe_number(fifty_two_week_high),
                    currency,
                )
                if not pd.isna(safe_number(fifty_two_week_high))
                else "-"
            ),
        )

        price_range_columns[1].metric(
            "52주 최저가",
            (
                format_money(
                    safe_number(fifty_two_week_low),
                    currency,
                )
                if not pd.isna(safe_number(fifty_two_week_low))
                else "-"
            ),
        )

        summary = (
            company_info.get("longBusinessSummary")
            or "기업 설명이 제공되지 않았습니다."
        )

        st.markdown("### 기업 개요")
        st.write(summary)

        company_details = {
            "웹사이트": company_info.get("website", "-"),
            "국가": company_info.get("country", "-"),
            "도시": company_info.get("city", "-"),
            "임직원 수": format_large_number(
                company_info.get("fullTimeEmployees")
            ),
            "산업": industry,
            "섹터": sector,
        }

        details_df = pd.DataFrame(
            company_details.items(),
            columns=["항목", "내용"],
        )

        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 탭 6: 원본 데이터
# =========================================================
with tabs[5]:
    st.markdown("### 조회된 주가 데이터")

    display_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA20",
        "SMA60",
        "RSI",
        "MACD",
        "Daily Return",
        "Drawdown",
    ]

    available_columns = [
        column for column in display_columns if column in df.columns
    ]

    table_data = df[available_columns].copy()
    table_data.index.name = "Date"

    styled_data = table_data.style.format(
        {
            "Open": "{:,.2f}",
            "High": "{:,.2f}",
            "Low": "{:,.2f}",
            "Close": "{:,.2f}",
            "Volume": "{:,.0f}",
            "SMA20": "{:,.2f}",
            "SMA60": "{:,.2f}",
            "RSI": "{:.2f}",
            "MACD": "{:.3f}",
            "Daily Return": "{:+.4%}",
            "Drawdown": "{:.2%}",
        },
        na_rep="-",
    )

    st.dataframe(
        styled_data,
        use_container_width=True,
        height=520,
    )

    csv_data = df.reset_index().to_csv(
        index=False,
        encoding="utf-8-sig",
    )

    st.download_button(
        label="📥 전체 데이터를 CSV로 다운로드",
        data=csv_data,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 하단 안내
# =========================================================
st.markdown("---")

st.caption(
    "⚠️ 본 대시보드는 교육 및 정보 제공을 위한 도구이며 투자 권유가 아닙니다. "
    "Yahoo Finance 데이터는 지연되거나 누락될 수 있으며, 실제 투자 판단 전에는 "
    "공식 거래소 또는 증권사 데이터를 확인해야 합니다."
)
