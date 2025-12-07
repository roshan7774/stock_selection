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


from fastapi import FastAPI, Response, status
import yfinance as yf
import pandas as pd
import requests
import numpy as np

app = FastAPI(title="Financial Data API", version="2.0")

def find_revenue_row(fin_df):
    if fin_df is None or fin_df.empty:
        return None
    for r in fin_df.index:
        rl = str(r).lower()
        if "revenue" in rl or "net sales" in rl:
            return r
    return None


def compute_3y_sales_growth_cagr(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    fin = t.financials

    if fin is None or fin.empty:
        return None

    row_name = find_revenue_row(fin)
    if not row_name:
        return t.info.get("revenueGrowth")

    revenues = fin.loc[row_name].dropna().astype(float)

    if len(revenues) >= 4:
        rev_t = revenues.iloc[-1]
        rev_t_minus_3 = revenues.iloc[-4]

        if rev_t_minus_3 <= 0:
            return None

        cagr = (rev_t / rev_t_minus_3) ** (1 / 3.0) - 1.0
        return float(cagr)

    elif len(revenues) >= 2:
        yoy = revenues.pct_change().dropna()
        return float(yoy.mean()) if not yoy.empty else None

    return t.info.get("revenueGrowth")


def compute_200_dma_signal(ticker_symbol):
    df = yf.download(ticker_symbol, period="1y", interval="1d")

    if df.empty or len(df) < 200:
        return None, None

    df["200_DMA"] = df["Close"].rolling(window=200).mean()

    current_price = float(df["Close"].iloc[-1])
    dma_200 = float(df["200_DMA"].iloc[-1])

    above_200_dma = current_price > dma_200

    return dma_200, above_200_dma


def get_best_possible_cagr_api(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    financials = ticker.financials

    if financials is None or financials.empty:
        return {"Revenue_CAGR": None, "Profit_CAGR": None}

    financials = financials.T.sort_index()

    try:
        revenue = financials["Total Revenue"]
        profit = financials["Net Income"]
    except KeyError:
        return {"Revenue_CAGR": None, "Profit_CAGR": None}

    def calculate_cagr(series):
        series = series.dropna()
        if len(series) < 2:
            return None
        start_val = series.iloc[0]
        end_val = series.iloc[-1]
        years = len(series) - 1
        if start_val <= 0:
            return None
        return float((end_val / start_val) ** (1 / years) - 1)

    return {
        "Revenue_CAGR": calculate_cagr(revenue) * 100 if calculate_cagr(revenue) else None,
        "Profit_CAGR": calculate_cagr(profit) * 100 if calculate_cagr(profit) else None
    }


# ------------------ ✅ NEW STOCK CALCULATOR (API SAFE) ------------------

class StockCalculator:
    def __init__(self, symbol):
        self.clean_symbol = symbol.upper().replace(".NS", "")
        self.yf_symbol = f"{self.clean_symbol}.NS"
        self.ticker = yf.Ticker(self.yf_symbol)

        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }
        self.session = requests.Session()

    def calculate_roe(self):
        try:
            fin = self.ticker.financials
            bs = self.ticker.balance_sheet

            net_income = fin.loc['Net Income'].iloc[0] if 'Net Income' in fin.index else None
            total_equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else None

            if net_income and total_equity:
                return round((net_income / total_equity) * 100, 2)
            return round(self.ticker.info.get('returnOnEquity', 0) * 100, 2)
        except:
            return None

    def calculate_de(self):
        try:
            bs = self.ticker.balance_sheet
            total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else None
            total_equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else None

            if total_debt and total_equity:
                return round(total_debt / total_equity, 2)

            ratio = self.ticker.info.get('debtToEquity')
            return round(ratio / 100, 2) if ratio else None
        except:
            return None

    def calculate_volume_spike(self):
        try:
            hist = self.ticker.history(period="2mo")
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].iloc[-21:-1].mean()
            if avg_volume == 0:
                return None
            return round(current_volume / avg_volume, 2)
        except:
            return None

    def get_delivery_nse_direct(self):
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=5)
            url = f"https://www.nseindia.com/api/quote-equity?symbol={self.clean_symbol}"
            r = self.session.get(url, headers=self.headers, timeout=5)

            if r.status_code == 200:
                data = r.json()
                return data.get('securityWiseDP', {}).get('deliveryToTradedQuantity')
            return None
        except:
            return None

    def get_promoter_holding(self):
        try:
            holders = self.ticker.major_holders
            if holders is not None and not holders.empty:
                for _, row in holders.iterrows():
                    if "Insiders" in str(row.iloc[1]):
                        return row.iloc[0]
            return self.ticker.info.get('heldPercentInsiders')
        except:
            return None


# ------------------ ✅ FINAL API ENDPOINT ------------------

@app.get("/financials/{ticker}", status_code=200)
def get_financial_data(ticker: str, response: Response):

    ticker_symbol = ticker.upper() + ".NS"

    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info

        sales_growth_3y = compute_3y_sales_growth_cagr(ticker_symbol)
        dma_200, dma_signal = compute_200_dma_signal(ticker_symbol)
        best_cagr_data = get_best_possible_cagr_api(ticker_symbol)

        calc = StockCalculator(ticker)

        data = {
            "Ticker": ticker_symbol,
            "DebtEquity": info.get("debtToEquity"),
            "Debt": info.get("totalDebt"),
            "SalesGrowth3Y": sales_growth_3y,
            "ProfitGrowth": info.get("earningsGrowth"),
            "EPS": info.get("trailingEps"),
            "DividendYield": info.get("dividendYield"),
            "BookValue": info.get("bookValue"),
            "PB": info.get("priceToBook"),
            "DMA_200": dma_200,
            "Above_200_DMA": dma_signal,
            "Revenue_CAGR": best_cagr_data["Revenue_CAGR"],
            "Profit_CAGR": best_cagr_data["Profit_CAGR"],
            "ROE": calc.calculate_roe(),
            "D_E": calc.calculate_de(),
            "Volume_Spike": calc.calculate_volume_spike(),
            "Delivery_Percentage": calc.get_delivery_nse_direct(),
            "Promoter_Holding": calc.get_promoter_holding()
        }

        return {"status": "success", "code": 200, "data": data}

    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"status": "error", "code": 400, "message": str(e)}
