from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# =========================================================
# Streamlit 기본 설정
# =========================================================
st.set_page_config(
    page_title="K-AI 반도체 주식 분석",
    page_icon="🇰🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 한국 AI·반도체 관련 종목
# =========================================================
# Yahoo Finance:
# 코스피 종목코드 + .KS
# 코스닥 종목코드 + .KQ

STOCK_GROUPS = {
    "AI 메모리·종합 반도체": {
        "삼성전자": {
            "ticker": "005930.KS",
            "description": "메모리·파운드리·시스템반도체",
        },
        "SK하이닉스": {
            "ticker": "000660.KS",
            "description": "HBM·DRAM·NAND 메모리",
        },
        "삼성전자우": {
            "ticker": "005935.KS",
            "description": "삼성전자 우선주",
        },
    },
    "반도체 장비": {
        "한미반도체": {
            "ticker": "042700.KS",
            "description": "HBM 관련 반도체 후공정 장비",
        },
        "원익IPS": {
            "ticker": "240810.KQ",
            "description": "반도체 증착·공정 장비",
        },
        "주성엔지니어링": {
            "ticker": "036930.KQ",
            "description": "반도체 증착 장비",
        },
        "테스": {
            "ticker": "095610.KQ",
            "description": "반도체 전공정 장비",
        },
        "유진테크": {
            "ticker": "084370.KQ",
            "description": "반도체 박막 증착 장비",
        },
        "피에스케이": {
            "ticker": "319660.KQ",
            "description": "반도체 세정·제거 장비",
        },
    },
    "반도체 검사·테스트": {
        "리노공업": {
            "ticker": "058470.KQ",
            "description": "반도체 테스트 핀·소켓",
        },
        "ISC": {
            "ticker": "095340.KQ",
            "description": "반도체 테스트 소켓",
        },
        "테크윙": {
            "ticker": "089030.KQ",
            "description": "반도체 검사·후공정 장비",
        },
        "고영": {
            "ticker": "098460.KQ",
            "description": "3D 검사장비·의료 로봇",
        },
    },
    "반도체 소재·부품": {
        "솔브레인": {
            "ticker": "357780.KQ",
            "description": "반도체 공정용 화학 소재",
        },
        "동진쎄미켐": {
            "ticker": "005290.KQ",
            "description": "포토레지스트·전자재료",
        },
        "이오테크닉스": {
            "ticker": "039030.KQ",
            "description": "반도체 레이저 장비",
        },
        "하나마이크론": {
            "ticker": "067310.KQ",
            "description": "반도체 패키징·테스트",
        },
        "심텍": {
            "ticker": "222800.KQ",
            "description": "반도체용 PCB·패키지 기판",
        },
    },
    "AI 소프트웨어·인프라": {
        "NAVER": {
            "ticker": "035420.KS",
            "description": "생성형 AI·클라우드·인터넷 플랫폼",
        },
        "카카오": {
            "ticker": "035720.KS",
            "description": "AI 서비스·인터넷 플랫폼",
        },
        "더존비즈온": {
            "ticker": "012510.KS",
            "description": "기업용 AI·클라우드 소프트웨어",
        },
        "한글과컴퓨터": {
            "ticker": "030520.KQ",
            "description": "AI 문서·소프트웨어",
        },
    },
}


BENCHMARKS = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
}


PERIODS = {
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "3년": "3y",
    "5년": "5y",
    "전체": "max",
}


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(37, 99, 235, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at top right,
                rgba(147, 51, 234, 0.12),
                transparent 27%
            );
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    .stock-header {
        padding: 18px 20px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        margin-bottom: 18px;
    }

    .stock-name {
        font-size: 1.7rem;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .stock-description {
        color: #94a3b8;
    }

    .positive-box {
        padding: 11px 13px;
        margin-bottom: 9px;
        border-radius: 11px;
        border: 1px solid rgba(34, 197, 94, 0.3);
        background: rgba(34, 197, 94, 0.12);
    }

    .negative-box {
        padding: 11px 13px;
        margin-bottom: 9px;
        border-radius: 11px;
        border: 1px solid rgba(239, 68, 68, 0.3);
        background: rgba(239, 68, 68, 0.12);
    }

    .neutral-box {
        padding: 11px 13px;
        margin-bottom: 9px;
        border-radius: 11px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        background: rgba(59, 130, 246, 0.12);
    }

    div[data-testid="stMetric"] {
        padding: 13px 15px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.035);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 데이터 함수
# =========================================================
def flatten_columns(data):
    """yfinance MultiIndex 컬럼을 일반 컬럼으로 변경합니다."""
    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)

        if "Close" in first_level:
            data.columns = first_level
        else:
            data.columns = data.columns.get_level_values(-1)

    return data


@st.cache_data(ttl=900, show_spinner=False)
def download_stock(ticker, period, start_date=None, end_date=None):
    """개별 종목 주가 데이터를 다운로드합니다."""
    try:
        if start_date is not None and end_date is not None:
            inclusive_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            data = yf.download(
                ticker,
                start=pd.Timestamp(start_date),
                end=inclusive_end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

        else:
            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

        if data is None or data.empty:
            return pd.DataFrame()

        data = flatten_columns(data.copy())
        data = data.loc[:, ~data.columns.duplicated()]
        data.index = pd.to_datetime(data.index)

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(column in data.columns for column in required):
            return pd.DataFrame()

        for column in required:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        return data.dropna(subset=["Close"])

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def download_multiple_stocks(tickers, period):
    """여러 종목의 종가를 한 번에 다운로드합니다."""
    try:
        ticker_list = list(dict.fromkeys(tickers))

        data = yf.download(
            ticker_list,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if len(ticker_list) == 1:
            data = flatten_columns(data)

            if "Close" not in data.columns:
                return pd.DataFrame()

            close = data[["Close"]].copy()
            close.columns = ticker_list

        else:
            if not isinstance(data.columns, pd.MultiIndex):
                return pd.DataFrame()

            if "Close" in data.columns.get_level_values(0):
                close = data["Close"].copy()

            elif "Close" in data.columns.get_level_values(1):
                close = data.xs(
                    "Close",
                    axis=1,
                    level=1,
                ).copy()

            else:
                return pd.DataFrame()

        close.index = pd.to_datetime(close.index)

        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)

        close = close.apply(pd.to_numeric, errors="coerce")

        return close.dropna(how="all")

    except Exception:
        return pd.DataFrame()


# =========================================================
# 기술지표 계산
# =========================================================
def calculate_rsi(close, period=14):
    difference = close.diff()

    gains = difference.clip(lower=0)
    losses = -difference.clip(upper=0)

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

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    return rsi.fillna(50)


def add_indicators(data):
    df = data.copy()

    df["Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (
        1 + df["Return"].fillna(0)
    ).cumprod() - 1

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()
    df["SMA120"] = df["Close"].rolling(120).mean()

    df["EMA12"] = df["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    df["EMA26"] = df["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["MACD Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["MACD Histogram"] = (
        df["MACD"] - df["MACD Signal"]
    )

    rolling_std = df["Close"].rolling(20).std()

    df["Bollinger Upper"] = (
        df["SMA20"] + 2 * rolling_std
    )

    df["Bollinger Lower"] = (
        df["SMA20"] - 2 * rolling_std
    )

    df["RSI"] = calculate_rsi(df["Close"])

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    cumulative_high = df["Close"].cummax()

    df["Drawdown"] = (
        df["Close"] / cumulative_high
    ) - 1

    return df


# =========================================================
# 유틸리티 함수
# =========================================================
def safe_float(value):
    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return np.nan

    except (TypeError, ValueError):
        return np.nan


def format_krw(value):
    value = safe_float(value)

    if pd.isna(value):
        return "-"

    return f"₩{value:,.0f}"


def format_volume(value):
    value = safe_float(value)

    if pd.isna(value):
        return "-"

    if abs(value) >= 100_000_000:
        return f"{value / 100_000_000:,.2f}억"

    if abs(value) >= 10_000:
        return f"{value / 10_000:,.1f}만"

    return f"{value:,.0f}"


def get_all_stock_names():
    names = []

    for group in STOCK_GROUPS.values():
        names.extend(group.keys())

    return names


def get_stock_info(stock_name):
    for category, stocks in STOCK_GROUPS.items():
        if stock_name in stocks:
            information = stocks[stock_name].copy()
            information["category"] = category
            return information

    return None


def ticker_to_name():
    result = {}

    for stocks in STOCK_GROUPS.values():
        for name, information in stocks.items():
            result[information["ticker"]] = name

    result["^KS11"] = "KOSPI"
    result["^KQ11"] = "KOSDAQ"

    return result


def create_signal_messages(df):
    latest = df.iloc[-1]
    messages = []

    close = safe_float(latest["Close"])
    sma20 = safe_float(latest["SMA20"])
    sma60 = safe_float(latest["SMA60"])
    rsi = safe_float(latest["RSI"])
    macd = safe_float(latest["MACD"])
    signal = safe_float(latest["MACD Signal"])

    if not pd.isna(sma20):
        if close > sma20:
            messages.append(
                (
                    "positive",
                    "현재 주가가 20일 이동평균선 위에 있습니다.",
                )
            )
        else:
            messages.append(
                (
                    "negative",
                    "현재 주가가 20일 이동평균선 아래에 있습니다.",
                )
            )

    if not pd.isna(sma20) and not pd.isna(sma60):
        if sma20 > sma60:
            messages.append(
                (
                    "positive",
                    "20일선이 60일선 위에 있어 단기 추세가 상대적으로 강합니다.",
                )
            )
        else:
            messages.append(
                (
                    "negative",
                    "20일선이 60일선 아래에 있어 단기 추세가 상대적으로 약합니다.",
                )
            )

    if not pd.isna(rsi):
        if rsi >= 70:
            messages.append(
                (
                    "negative",
                    f"RSI가 {rsi:.1f}로 과매수 구간에 있습니다.",
                )
            )
        elif rsi <= 30:
            messages.append(
                (
                    "neutral",
                    f"RSI가 {rsi:.1f}로 과매도 구간에 있습니다.",
                )
            )
        else:
            messages.append(
                (
                    "neutral",
                    f"RSI가 {rsi:.1f}로 중립 구간에 있습니다.",
                )
            )

    if not pd.isna(macd) and not pd.isna(signal):
        if macd > signal:
            messages.append(
                (
                    "positive",
                    "MACD가 시그널선 위에 있습니다.",
                )
            )
        else:
            messages.append(
                (
                    "negative",
                    "MACD가 시그널선 아래에 있습니다.",
                )
            )

    return messages


# =========================================================
# 가격 차트 함수
# =========================================================
def make_price_chart(
    df,
    stock_name,
    show_sma20,
    show_sma60,
    show_sma120,
    show_bollinger,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=stock_name,
            increasing_line_color="#ef4444",
            decreasing_line_color="#3b82f6",
        ),
        row=1,
        col=1,
    )

    if show_sma20:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA20"],
                name="20일선",
                mode="lines",
                line=dict(
                    width=1.5,
                    color="#f59e0b",
                ),
            ),
            row=1,
            col=1,
        )

    if show_sma60:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA60"],
                name="60일선",
                mode="lines",
                line=dict(
                    width=1.5,
                    color="#a855f7",
                ),
            ),
            row=1,
            col=1,
        )

    if show_sma120:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA120"],
                name="120일선",
                mode="lines",
                line=dict(
                    width=1.5,
                    color="#14b8a6",
                ),
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
                line=dict(
                    width=1,
                    dash="dot",
                    color="rgba(148,163,184,0.7)",
                ),
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
                line=dict(
                    width=1,
                    dash="dot",
                    color="rgba(148,163,184,0.7)",
                ),
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.08)",
            ),
            row=1,
            col=1,
        )

    volume_colors = np.where(
        df["Close"] >= df["Open"],
        "rgba(239,68,68,0.7)",
        "rgba(59,130,246,0.7)",
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="거래량",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{stock_name} 주가와 거래량",
        template="plotly_dark",
        height=690,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20,
        ),
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
        rangeselector=dict(
            buttons=[
                dict(
                    count=1,
                    label="1M",
                    step="month",
                    stepmode="backward",
                ),
                dict(
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward",
                ),
                dict(
                    count=6,
                    label="6M",
                    step="month",
                    stepmode="backward",
                ),
                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward",
                ),
                dict(
                    step="all",
                    label="ALL",
                ),
            ],
            bgcolor="rgba(30,41,59,0.85)",
            activecolor="#2563eb",
            font=dict(color="white"),
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="가격",
        gridcolor="rgba(148,163,184,0.10)",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="거래량",
        gridcolor="rgba(148,163,184,0.10)",
        row=2,
        col=1,
    )

    return fig


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.markdown("## 🇰🇷 종목 설정")

    selected_category = st.selectbox(
        "산업군",
        list(STOCK_GROUPS.keys()),
    )

    category_stocks = STOCK_GROUPS[selected_category]

    selected_stock = st.selectbox(
        "분석 종목",
        list(category_stocks.keys()),
    )

    stock_information = category_stocks[selected_stock]
    selected_ticker = stock_information["ticker"]

    st.caption(
        f"{stock_information['description']}\n\n"
        f"Yahoo Finance 티커: {selected_ticker}"
    )

    st.markdown("---")

    date_setting = st.radio(
        "조회 기간",
        ["간편 선택", "날짜 직접 선택"],
        horizontal=True,
    )

    if date_setting == "간편 선택":
        selected_period_name = st.selectbox(
            "기간",
            list(PERIODS.keys()),
            index=2,
        )

        selected_period = PERIODS[selected_period_name]
        start_date = None
        end_date = None

    else:
        end_date = st.date_input(
            "종료일",
            value=date.today(),
            max_value=date.today(),
        )

        start_date = st.date_input(
            "시작일",
            value=end_date - timedelta(days=365),
            max_value=end_date,
        )

        selected_period = "1y"

    st.markdown("---")
    st.markdown("### 차트 지표")

    show_sma20 = st.checkbox(
        "20일 이동평균선",
        value=True,
    )

    show_sma60 = st.checkbox(
        "60일 이동평균선",
        value=True,
    )

    show_sma120 = st.checkbox(
        "120일 이동평균선",
        value=False,
    )

    show_bollinger = st.checkbox(
        "볼린저 밴드",
        value=True,
    )

    st.markdown("---")

    if st.button(
        "🔄 데이터 새로고침",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Yahoo Finance의 지연 데이터를 활용합니다. "
        "실제 주문 전에는 증권사 데이터를 확인하세요."
    )


# =========================================================
# 메인 화면
# =========================================================
st.markdown(
    '<div class="main-title">🇰🇷 K-AI 반도체 주식 분석</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-subtitle">
    한국 AI·반도체 대표 종목의 주가, 기술지표, 위험도와
    업종 내 상대 성과를 분석합니다.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.spinner(
    f"{selected_stock} 주가 데이터를 불러오는 중입니다..."
):
    raw_data = download_stock(
        selected_ticker,
        selected_period,
        start_date,
        end_date,
    )


if raw_data.empty:
    st.error(
        "주가 데이터를 불러오지 못했습니다. "
        "Yahoo Finance의 일시적인 접속 제한일 수 있습니다. "
        "잠시 후 데이터 새로고침 버튼을 눌러 주세요."
    )
    st.stop()


df = add_indicators(raw_data)


st.markdown(
    f"""
    <div class="stock-header">
        <div class="stock-name">
            {selected_stock}
        </div>
        <div class="stock-description">
            {selected_category} ·
            {selected_ticker} ·
            {stock_information["description"]}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 핵심 지표
# =========================================================
latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) >= 2 else latest

latest_price = safe_float(latest["Close"])
previous_price = safe_float(previous["Close"])

daily_change = latest_price - previous_price

daily_change_rate = (
    daily_change / previous_price * 100
    if previous_price != 0
    else 0
)

first_price = safe_float(df["Close"].iloc[0])

period_return = (
    (latest_price / first_price - 1) * 100
    if first_price != 0
    else np.nan
)

period_high = safe_float(df["High"].max())
period_low = safe_float(df["Low"].min())

daily_returns = df["Return"].dropna()

volatility = (
    daily_returns.std() * np.sqrt(252) * 100
    if len(daily_returns) > 1
    else np.nan
)

maximum_drawdown = safe_float(
    df["Drawdown"].min()
) * 100


metric_columns = st.columns(6)

metric_columns[0].metric(
    "현재 가격",
    format_krw(latest_price),
    f"{daily_change_rate:+.2f}%",
)

metric_columns[1].metric(
    "조회 기간 수익률",
    f"{period_return:+.2f}%",
)

metric_columns[2].metric(
    "기간 최고가",
    format_krw(period_high),
)

metric_columns[3].metric(
    "기간 최저가",
    format_krw(period_low),
)

metric_columns[4].metric(
    "연환산 변동성",
    (
        f"{volatility:.2f}%"
        if not pd.isna(volatility)
        else "-"
    ),
)

metric_columns[5].metric(
    "최대 낙폭",
    (
        f"{maximum_drawdown:.2f}%"
        if not pd.isna(maximum_drawdown)
        else "-"
    ),
)


# =========================================================
# 탭
# =========================================================
tabs = st.tabs(
    [
        "📈 주가 차트",
        "🧭 기술적 분석",
        "⚖️ 업종 종목 비교",
        "📊 위험 분석",
        "🗂️ 데이터",
    ]
)


# =========================================================
# 탭 1: 가격 차트
# =========================================================
with tabs[0]:
    price_chart = make_price_chart(
        df,
        selected_stock,
        show_sma20,
        show_sma60,
        show_sma120,
        show_bollinger,
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    detail_columns = st.columns(5)

    detail_columns[0].metric(
        "시가",
        format_krw(latest["Open"]),
    )

    detail_columns[1].metric(
        "고가",
        format_krw(latest["High"]),
    )

    detail_columns[2].metric(
        "저가",
        format_krw(latest["Low"]),
    )

    detail_columns[3].metric(
        "종가",
        format_krw(latest["Close"]),
    )

    detail_columns[4].metric(
        "거래량",
        format_volume(latest["Volume"]),
    )


# =========================================================
# 탭 2: 기술적 분석
# =========================================================
with tabs[1]:
    left_column, right_column = st.columns(
        [0.35, 0.65]
    )

    with left_column:
        st.markdown("### 지표 신호")

        st.caption(
            "이 신호는 단순 기술지표 규칙에 따른 "
            "참고 정보이며 매수·매도 추천이 아닙니다."
        )

        for signal_type, message in create_signal_messages(df):
            if signal_type == "positive":
                box_class = "positive-box"
                icon = "🟢"

            elif signal_type == "negative":
                box_class = "negative-box"
                icon = "🔴"

            else:
                box_class = "neutral-box"
                icon = "🔵"

            st.markdown(
                f"""
                <div class="{box_class}">
                    {icon} {message}
                </div>
                """,
                unsafe_allow_html=True,
            )

        latest_rsi = safe_float(latest["RSI"])
        latest_atr = safe_float(latest["ATR14"])

        indicator_columns = st.columns(2)

        indicator_columns[0].metric(
            "RSI(14)",
            f"{latest_rsi:.2f}",
        )

        indicator_columns[1].metric(
            "ATR(14)",
            format_krw(latest_atr),
        )

        indicator_columns[0].metric(
            "MACD",
            f"{safe_float(latest['MACD']):.2f}",
        )

        indicator_columns[1].metric(
            "MACD Signal",
            f"{safe_float(latest['MACD Signal']):.2f}",
        )

    with right_column:
        rsi_chart = go.Figure()

        rsi_chart.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(
                    width=2,
                    color="#60a5fa",
                ),
            )
        )

        rsi_chart.add_hline(
            y=70,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="과매수 70",
        )

        rsi_chart.add_hline(
            y=30,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text="과매도 30",
        )

        rsi_chart.update_layout(
            title="RSI 상대강도지수",
            template="plotly_dark",
            height=360,
            yaxis=dict(range=[0, 100]),
            hovermode="x unified",
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            rsi_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )

    macd_colors = np.where(
        df["MACD Histogram"] >= 0,
        "rgba(239,68,68,0.75)",
        "rgba(59,130,246,0.75)",
    )

    macd_chart = go.Figure()

    macd_chart.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD Histogram"],
            name="히스토그램",
            marker_color=macd_colors,
        )
    )

    macd_chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            mode="lines",
            line=dict(
                width=2,
                color="#60a5fa",
            ),
        )
    )

    macd_chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD Signal"],
            name="Signal",
            mode="lines",
            line=dict(
                width=1.8,
                color="#f59e0b",
            ),
        )
    )

    macd_chart.add_hline(
        y=0,
        line_color="rgba(148,163,184,0.6)",
    )

    macd_chart.update_layout(
        title="MACD",
        template="plotly_dark",
        height=410,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        legend=dict(orientation="h"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        macd_chart,
        use_container_width=True,
        config={"displaylogo": False},
    )


# =========================================================
# 탭 3: 산업군 비교
# =========================================================
with tabs[2]:
    st.markdown(
        f"### {selected_category} 종목 상대 성과"
    )

    comparison_options = list(category_stocks.keys())

    default_selection = comparison_options[:5]

    if selected_stock not in default_selection:
        default_selection = [
            selected_stock
        ] + default_selection[:4]

    selected_comparison_names = st.multiselect(
        "비교 종목",
        comparison_options,
        default=default_selection,
        max_selections=8,
    )

    include_kospi = st.checkbox(
        "KOSPI 지수 함께 비교",
        value=True,
    )

    include_kosdaq = st.checkbox(
        "KOSDAQ 지수 함께 비교",
        value=False,
    )

    comparison_period_name = st.selectbox(
        "비교 기간",
        ["3개월", "6개월", "1년", "2년", "3년", "5년"],
        index=2,
    )

    comparison_tickers = [
        category_stocks[name]["ticker"]
        for name in selected_comparison_names
    ]

    if include_kospi:
        comparison_tickers.append("^KS11")

    if include_kosdaq:
        comparison_tickers.append("^KQ11")

    if len(comparison_tickers) == 0:
        st.info("비교할 종목을 한 개 이상 선택해 주세요.")

    else:
        comparison_data = download_multiple_stocks(
            tuple(comparison_tickers),
            PERIODS[comparison_period_name],
        )

        if comparison_data.empty:
            st.warning(
                "비교 데이터를 불러오지 못했습니다."
            )

        else:
            name_mapping = ticker_to_name()

            comparison_data = comparison_data.rename(
                columns=name_mapping
            )

            normalized = pd.DataFrame(
                index=comparison_data.index
            )

            for column in comparison_data.columns:
                valid_values = comparison_data[
                    column
                ].dropna()

                if (
                    not valid_values.empty
                    and valid_values.iloc[0] != 0
                ):
                    normalized[column] = (
                        comparison_data[column]
                        / valid_values.iloc[0]
                        * 100
                    )

            comparison_chart = go.Figure()

            for column in normalized.columns:
                comparison_chart.add_trace(
                    go.Scatter(
                        x=normalized.index,
                        y=normalized[column],
                        name=column,
                        mode="lines",
                        line=dict(width=2),
                    )
                )

            comparison_chart.add_hline(
                y=100,
                line_dash="dash",
                line_color="rgba(148,163,184,0.6)",
            )

            comparison_chart.update_layout(
                title="시작일을 100으로 환산한 상대 수익률",
                template="plotly_dark",
                height=570,
                hovermode="x unified",
                xaxis_title="날짜",
                yaxis_title="정규화 가격",
                margin=dict(
                    l=20,
                    r=20,
                    t=65,
                    b=20,
                ),
                legend=dict(orientation="h"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                comparison_chart,
                use_container_width=True,
                config={"displaylogo": False},
            )

            comparison_summary = []

            for column in comparison_data.columns:
                prices = comparison_data[column].dropna()

                if len(prices) < 2:
                    continue

                stock_returns = prices.pct_change().dropna()

                cumulative_return = (
                    prices.iloc[-1]
                    / prices.iloc[0]
                    - 1
                ) * 100

                stock_volatility = (
                    stock_returns.std()
                    * np.sqrt(252)
                    * 100
                )

                peak = prices.cummax()

                stock_drawdown = (
                    prices / peak - 1
                ).min() * 100

                comparison_summary.append(
                    {
                        "종목": column,
                        "기간 수익률(%)": cumulative_return,
                        "연환산 변동성(%)": stock_volatility,
                        "최대 낙폭(%)": stock_drawdown,
                    }
                )

            summary_df = pd.DataFrame(
                comparison_summary
            )

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
                            "최대 낙폭(%)": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            return_data = (
                comparison_data
                .pct_change()
                .dropna(how="all")
            )

            if len(return_data.columns) >= 2:
                correlation = return_data.corr()

                heatmap = go.Figure(
                    data=go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns,
                        y=correlation.index,
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu",
                        reversescale=True,
                        text=np.round(
                            correlation.values,
                            2,
                        ),
                        texttemplate="%{text}",
                        hovertemplate=(
                            "%{x} / %{y}"
                            "<br>상관계수: %{z:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                heatmap.update_layout(
                    title="일간 수익률 상관관계",
                    template="plotly_dark",
                    height=520,
                    margin=dict(
                        l=20,
                        r=20,
                        t=65,
                        b=20,
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    heatmap,
                    use_container_width=True,
                    config={"displaylogo": False},
                )


# =========================================================
# 탭 4: 위험 분석
# =========================================================
with tabs[3]:
    chart_left, chart_right = st.columns(2)

    with chart_left:
        return_histogram = go.Figure()

        return_histogram.add_trace(
            go.Histogram(
                x=daily_returns * 100,
                nbinsx=50,
                marker_color="#60a5fa",
                opacity=0.85,
                name="일간 수익률",
            )
        )

        return_histogram.add_vline(
            x=0,
            line_dash="dash",
            line_color="white",
        )

        return_histogram.update_layout(
            title="일간 수익률 분포",
            template="plotly_dark",
            height=420,
            xaxis_title="수익률(%)",
            yaxis_title="빈도",
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            return_histogram,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with chart_right:
        cumulative_chart = go.Figure()

        cumulative_chart.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cumulative Return"] * 100,
                name="누적 수익률",
                mode="lines",
                fill="tozeroy",
                line=dict(
                    width=2,
                    color="#ef4444",
                ),
                fillcolor="rgba(239,68,68,0.12)",
            )
        )

        cumulative_chart.add_hline(
            y=0,
            line_dash="dash",
            line_color="rgba(148,163,184,0.6)",
        )

        cumulative_chart.update_layout(
            title="누적 수익률",
            template="plotly_dark",
            height=420,
            xaxis_title="날짜",
            yaxis_title="수익률(%)",
            hovermode="x unified",
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            cumulative_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )

    drawdown_chart = go.Figure()

    drawdown_chart.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown"] * 100,
            name="낙폭",
            mode="lines",
            fill="tozeroy",
            line=dict(
                width=2,
                color="#3b82f6",
            ),
            fillcolor="rgba(59,130,246,0.15)",
        )
    )

    drawdown_chart.update_layout(
        title="고점 대비 낙폭",
        template="plotly_dark",
        height=410,
        xaxis_title="날짜",
        yaxis_title="낙폭(%)",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        drawdown_chart,
        use_container_width=True,
        config={"displaylogo": False},
    )

    if len(daily_returns) > 1:
        positive_ratio = (
            daily_returns > 0
        ).mean() * 100

        average_return = (
            daily_returns.mean() * 100
        )

        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100

        downside_returns = daily_returns[
            daily_returns < 0
        ]

        downside_volatility = (
            downside_returns.std()
            * np.sqrt(252)
            if len(downside_returns) > 1
            else np.nan
        )

        annualized_return = (
            daily_returns.mean() * 252
        )

        sortino_ratio = (
            annualized_return / downside_volatility
            if (
                not pd.isna(downside_volatility)
                and downside_volatility != 0
            )
            else np.nan
        )

        risk_columns = st.columns(5)

        risk_columns[0].metric(
            "상승일 비율",
            f"{positive_ratio:.1f}%",
        )

        risk_columns[1].metric(
            "평균 일간 수익률",
            f"{average_return:+.3f}%",
        )

        risk_columns[2].metric(
            "최고 일간 수익률",
            f"{best_day:+.2f}%",
        )

        risk_columns[3].metric(
            "최저 일간 수익률",
            f"{worst_day:+.2f}%",
        )

        risk_columns[4].metric(
            "Sortino Ratio",
            (
                f"{sortino_ratio:.2f}"
                if not pd.isna(sortino_ratio)
                else "-"
            ),
        )


# =========================================================
# 탭 5: 데이터
# =========================================================
with tabs[4]:
    st.markdown("### 주가와 기술지표 데이터")

    display_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA20",
        "SMA60",
        "SMA120",
        "RSI",
        "MACD",
        "MACD Signal",
        "Return",
        "Drawdown",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    table_data = df[
        available_columns
    ].copy()

    table_data.index.name = "Date"

    st.dataframe(
        table_data.style.format(
            {
                "Open": "{:,.0f}",
                "High": "{:,.0f}",
                "Low": "{:,.0f}",
                "Close": "{:,.0f}",
                "Volume": "{:,.0f}",
                "SMA20": "{:,.0f}",
                "SMA60": "{:,.0f}",
                "SMA120": "{:,.0f}",
                "RSI": "{:.2f}",
                "MACD": "{:.2f}",
                "MACD Signal": "{:.2f}",
                "Return": "{:+.2%}",
                "Drawdown": "{:.2%}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        height=550,
    )

    csv_data = df.reset_index().to_csv(
        index=False,
        encoding="utf-8-sig",
    )

    st.download_button(
        "📥 CSV 데이터 다운로드",
        data=csv_data,
        file_name=(
            f"{selected_stock}_{selected_ticker}_stock_data.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 하단 안내
# =========================================================
st.markdown("---")

st.caption(
    "⚠️ 본 웹앱은 교육 및 데이터 분석 목적으로 제작되었습니다. "
    "표시되는 기술적 지표와 자동 신호는 투자 추천이 아닙니다. "
    "Yahoo Finance 데이터는 지연되거나 일시적으로 누락될 수 있습니다."
)
