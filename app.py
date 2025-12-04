# # main.py
# import requests
# from typing import Any, Dict, Optional
# from fastapi import FastAPI, Query
# from pydantic import BaseModel
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import traceback
# import logging

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("financial_api")

# app = FastAPI(title="Financial Data API", version="3.1 – Stable No-NSEpython")


# # ------------------------
# # Helper Cleaners
# # ------------------------
# def clean(val: Any) -> Any:
#     if val is None:
#         return None
#     if isinstance(val, (np.integer, np.int64, np.int32)):
#         return int(val)
#     if isinstance(val, (np.floating, np.float32, np.float64)):
#         return float(val)
#     try:
#         if pd.isna(val):
#             return None
#     except Exception:
#         pass
#     return val


# def try_parse_series_as_float(series: pd.Series) -> pd.Series:
#     try:
#         return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")
#     except Exception:
#         return series


# def to_float_safe(v: Any) -> Optional[float]:
#     try:
#         if v is None:
#             return None
#         s = str(v).replace(",", "")
#         if s.startswith("(") and s.endswith(")"):
#             s = "-" + s[1:-1]
#         return float(s)
#     except Exception:
#         return None


# # ------------------------
# # CAGR Calculation
# # ------------------------
# def compute_cagr(series: Optional[pd.Series], years: int = 5) -> Optional[float]:
#     if series is None:
#         return None
#     try:
#         parsed_idx = pd.to_datetime(series.index, errors="coerce")
#         if not parsed_idx.isna().all():
#             tmp = pd.Series(series.values, index=parsed_idx).sort_index()
#         else:
#             tmp = series.copy()

#         tmp = tmp.dropna().astype(float)

#         if len(tmp) < (years + 1):
#             return None

#         start = to_float_safe(tmp.iloc[-(years + 1)])
#         end = to_float_safe(tmp.iloc[-1])

#         if not start or not end or start <= 0 or end <= 0:
#             return None

#         cagr = (end / start) ** (1 / years) - 1
#         return round(cagr * 100, 2)
#     except Exception:
#         return None


# # ------------------------
# # NSE Annual Reports Fetcher (Stable)
# # ------------------------
# def fetch_nse_annual_reports(symbol: str) -> Optional[pd.DataFrame]:
#     """
#     Safe version — detects HTML (blocked) and avoids crashes.
#     """
#     s = symbol.upper().replace(".NS", "").replace(".BO", "")
#     url = f"https://www.nseindia.com/api/annual-reports?symbol={s}"

#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
#         "Accept": "application/json, text/plain, */*",
#         "Accept-Language": "en-US,en;q=0.9",
#         "Referer": "https://www.nseindia.com/",
#         "Connection": "keep-alive",
#     }

#     session = requests.Session()

#     # Step 1: Get cookies
#     try:
#         session.get("https://www.nseindia.com", headers=headers, timeout=8)
#     except Exception:
#         logger.warning("Failed initial NSE homepage cookie setup.")

#     # Step 2: Fetch annual reports JSON
#     try:
#         resp = session.get(url, headers=headers, timeout=10)

#         # # Detect anti-bot HTML
#         # if "text/html" in resp.headers.get("Content-Type", ""):
#         #     logger.error(f"NSE blocked request for {s} (HTML returned).")
#         #     return None

#         # Attempt JSON
#         try:
#             payload = resp.json()
#         except Exception as e:
#             logger.error(f"Invalid JSON received for {s}. Error : {e}")
#             return None

#         # Extract data array
#         data_arr = (
#             payload.get("annualReports")
#             or payload.get("data")
#             or (payload if isinstance(payload, list) else None)
#         )

#         if not data_arr:
#             return None

#         df = pd.DataFrame(data_arr)

#         # Index by year-like column
#         for col in ["year", "fy", "financialYear"]:
#             if col in df.columns:
#                 df = df.set_index(col)
#                 break

#         return df

#     except Exception:
#         logger.exception("Failed to fetch NSE annual reports.")
#         return None


# # ------------------------
# # Extract revenue & profit
# # ------------------------
# def extract_revenue_profit_from_nse(df: pd.DataFrame):
#     if df is None or df.empty:
#         return None, None

#     rev_col = None
#     prof_col = None

#     for c in df.columns:
#         l = c.lower()
#         if "revenue" in l or "net sales" in l or "net_sales" in l:
#             rev_col = c
#         if "profit" in l or "pat" in l:
#             prof_col = c

#     rev = try_parse_series_as_float(df[rev_col]) if rev_col else None
#     prof = try_parse_series_as_float(df[prof_col]) if prof_col else None

#     return rev, prof


# # ------------------------
# # yfinance technicals
# # ------------------------
# def get_yf_technicals(symbol: str):
#     try:
#         t = yf.Ticker(symbol)
#         hist = t.history(period="1y", auto_adjust=False)

#         if hist is None or hist.empty:
#             return None, None, None

#         hist = hist.sort_index()

#         ma200 = None
#         above200 = None
#         vol_spike = None

#         if "Close" in hist and len(hist["Close"].dropna()) >= 200:
#             ma200 = float(hist["Close"].rolling(200).mean().iloc[-1])
#             latest_close = float(hist["Close"].iloc[-1])
#             above200 = latest_close > ma200

#         if "Volume" in hist and len(hist["Volume"].dropna()) >= 21:
#             avg20 = hist["Volume"].tail(21).iloc[:-1].mean()
#             latest_vol = hist["Volume"].iloc[-1]
#             if avg20 > 0:
#                 vol_spike = float(latest_vol / avg20)

#         return ma200, above200, vol_spike
#     except Exception:
#         logger.exception("yfinance technicals failed.")
#         return None, None, None


# # ------------------------
# # API Response Model
# # ------------------------
# class ApiResponse(BaseModel):
#     status: str
#     data: Dict[str, Any]


# # ------------------------
# # Main Endpoint
# # ------------------------
# @app.get("/financials/{ticker}", response_model=ApiResponse)
# def get_financials(ticker: str):
#     symbol = ticker.upper()
#     if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
#         symbol += ".NS"

#     try:
#         t = yf.Ticker(symbol)

#         try:
#             info = t.info or {}
#         except Exception:
#             info = {}

#         # Fetch NSE annual reports (for CAGR)
#         df_nse = fetch_nse_annual_reports(symbol)
#         rev_cagr_5y = None
#         prof_cagr_5y = None

#         if df_nse is not None:
#             rev_series, prof_series = extract_revenue_profit_from_nse(df_nse)
#             rev_cagr_5y = compute_cagr(rev_series, 5)
#             prof_cagr_5y = compute_cagr(prof_series, 5)

#         # Technical indicators
#         ma200, above200, volspike = get_yf_technicals(symbol)

#         data = {
#             "Ticker": symbol,
#             "DebtEquity": clean(info.get("debtToEquity")),
#             "Debt": clean(info.get("totalDebt")),
#             "PromoterHolding": clean(info.get("heldPercentInsiders")),
#             "ROE": clean(info.get("returnOnEquity")),
#             "BookValue": clean(info.get("bookValue")),
#             "PB": clean(info.get("priceToBook")),
#             "5YRevenueCAGR": clean(rev_cagr_5y),
#             "5YProfitCAGR": clean(prof_cagr_5y),
#             "ProfitGrowth": clean(info.get("earningsGrowth")),
#             "EPS": clean(info.get("trailingEps")),
#             "DividendYield": clean(info.get("dividendYield")),
#             "Sector": clean(info.get("sector")),
#             "200_DMA": clean(ma200),
#             "Above200DMA": clean(above200),
#             "VolumeSpikeRatio": clean(volspike),
#         }

#         return {"status": "success", "data": data}

#     except Exception as e:
#         tb = traceback.format_exc()
#         logger.exception("Unhandled exception in /financials")
#         return {"status": "error", "message": str(e), "trace": tb}



# main.py
import requests
from typing import Any, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial_api")

app = FastAPI(title="Financial Data API", version="3.2 – Stable with 3Y Sales Growth")


# ------------------------
# Utility Cleaners
# ------------------------
def clean(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


def try_parse_series_as_float(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")
    except Exception:
        return series


def to_float_safe(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).replace(",", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        return float(s)
    except Exception:
        return None


# ------------------------
# CAGR Calculator (Generic)
# ------------------------
def compute_cagr(series: Optional[pd.Series], years: int = 5) -> Optional[float]:
    if series is None:
        return None
    try:
        parsed_idx = pd.to_datetime(series.index, errors="coerce")
        if not parsed_idx.isna().all():
            tmp = pd.Series(series.values, index=parsed_idx).sort_index()
        else:
            tmp = series.copy()

        tmp = tmp.dropna().astype(float)

        if len(tmp) < (years + 1):
            return None

        start = to_float_safe(tmp.iloc[-(years + 1)])
        end = to_float_safe(tmp.iloc[-1])

        if not start or not end or start <= 0 or end <= 0:
            return None

        cagr = (end / start) ** (1 / years) - 1
        return round(cagr * 100, 2)

    except Exception:
        return None


# ------------------------
# find revenue row in yfinance financials
# ------------------------
def find_revenue_row(fin_df):
    if fin_df is None or fin_df.empty:
        return None
    for pattern in (
        ("total", "revenue"),
        ("total revenue",),
        ("totalrevenue",),
        ("revenue",),
        ("net sales",)
    ):
        for r in fin_df.index:
            rl = str(r).lower()
            if all(p in rl for p in pattern):
                return r
    return None


# ------------------------
# Compute 3-Year Sales Growth CAGR using yfinance
# ------------------------
def compute_3y_sales_growth_cagr(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    fin = t.financials

    if fin is None or fin.empty:
        try:
            fin = t.get_financials()
        except AttributeError:
            fin = pd.DataFrame()

    row_name = find_revenue_row(fin)

    if not row_name:
        info = t.info
        return info.get("revenueGrowth")

    revenues = fin.loc[row_name].dropna().astype(float)

    if revenues.empty:
        return None

    try:
        cols_sorted = sorted(revenues.index, key=lambda x: pd.to_datetime(x))
    except Exception:
        cols_sorted = list(revenues.index)

    revenues = revenues[cols_sorted]

    if len(revenues) >= 4:
        rev_t = revenues.iloc[-1]
        rev_t_3 = revenues.iloc[-4]

        if rev_t_3 <= 0 or pd.isna(rev_t_3) or pd.isna(rev_t):
            return None

        cagr = (rev_t / rev_t_3) ** (1/3) - 1
        return round(cagr * 100, 2)

    elif len(revenues) >= 2:
        yoy = revenues.pct_change().dropna()
        if yoy.empty:
            return None
        return round(float(yoy.mean()) * 100, 2)

    else:
        info = t.info
        return info.get("revenueGrowth")


# ------------------------
# Extract revenue & profit from NSE Annual Reports
# ------------------------
def fetch_nse_annual_reports(symbol: str) -> Optional[pd.DataFrame]:
    s = symbol.upper().replace(".NS", "").replace(".BO", "")
    url = f"https://www.nseindia.com/api/annual-reports?symbol={s}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/"
    }

    session = requests.Session()

    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
    except Exception:
        pass

    try:
        resp = session.get(url, headers=headers, timeout=10)
        payload = resp.json()

        data_arr = payload.get("annualReports") or payload.get("data") or payload

        if not data_arr:
            return None

        df = pd.DataFrame(data_arr)

        for col in ["year", "fy", "financialYear"]:
            if col in df.columns:
                df = df.set_index(col)
                break

        return df

    except Exception:
        return None


def extract_revenue_profit_from_nse(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None

    rev_col = None
    prof_col = None

    for c in df.columns:
        l = c.lower()
        if "revenue" in l or "net sales" in l:
            rev_col = c
        if "profit" in l or "pat" in l:
            prof_col = c

    rev = try_parse_series_as_float(df[rev_col]) if rev_col else None
    prof = try_parse_series_as_float(df[prof_col]) if prof_col else None

    return rev, prof


# ------------------------
# Technical indicators (MA200, Volume spike)
# ------------------------
def get_yf_technicals(symbol: str):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y")

        if hist is None or hist.empty:
            return None, None, None

        hist = hist.sort_index()

        ma200, above200, vol_spike = None, None, None

        if len(hist["Close"]) >= 200:
            ma200 = float(hist["Close"].rolling(200).mean().iloc[-1])
            above200 = float(hist["Close"].iloc[-1]) > ma200

        if len(hist["Volume"]) >= 21:
            avg20 = hist["Volume"].tail(21).iloc[:-1].mean()
            latest = hist["Volume"].iloc[-1]
            if avg20 > 0:
                vol_spike = latest / avg20

        return ma200, above200, vol_spike

    except Exception:
        return None, None, None


# ------------------------
# API Model
# ------------------------
class ApiResponse(BaseModel):
    status: str
    data: Dict[str, Any]


# ------------------------
# MAIN ENDPOINT
# ------------------------
@app.get("/financials/{ticker}", response_model=ApiResponse)
def get_financials(ticker: str):
    symbol = ticker.upper()
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        symbol += ".NS"

    try:
        t = yf.Ticker(symbol)

        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Fetch NSE Data
        df_nse = fetch_nse_annual_reports(symbol)
        rev_series, prof_series = extract_revenue_profit_from_nse(df_nse) if df_nse is not None else (None, None)

        rev_cagr_5y = compute_cagr(rev_series, 5)
        prof_cagr_5y = compute_cagr(prof_series, 5)

        # Technicals
        ma200, above200, volspike = get_yf_technicals(symbol)

        # NEW FIELD from your first script
        sales_growth_3y = compute_3y_sales_growth_cagr(symbol)

        data = {
            "Ticker": symbol,
            "DebtEquity": clean(info.get("debtToEquity")),
            "Debt": clean(info.get("totalDebt")),
            "PromoterHolding": clean(info.get("heldPercentInsiders")),
            "ROE": clean(info.get("returnOnEquity")),
            "BookValue": clean(info.get("bookValue")),
            "PB": clean(info.get("priceToBook")),

            # CAGR metrics
            "5YRevenueCAGR": clean(rev_cagr_5y),
            "5YProfitCAGR": clean(prof_cagr_5y),

            # NEW FIELD ADDED HERE
            "SalesGrowth3Y": clean(sales_growth_3y),

            "ProfitGrowth": clean(info.get("earningsGrowth")),
            "EPS": clean(info.get("trailingEps")),
            "DividendYield": clean(info.get("dividendYield")),
            "Sector": clean(info.get("sector")),

            # Technicals
            "200_DMA": clean(ma200),
            "Above200DMA": clean(above200),
            "VolumeSpikeRatio": clean(volspike),
        }

        return {"status": "success", "data": data}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Exception in /financials")
        return {"status": "error", "message": str(e), "trace": tb}
