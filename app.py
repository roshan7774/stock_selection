from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import math

app = FastAPI(title="Financial Data API", version="1.0")


def find_revenue_row(fin_df):
    """Try several heuristics to locate the revenue row name in the financials DataFrame."""
    if fin_df is None or fin_df.empty:
        return None
    rows = [r.lower() for r in fin_df.index.astype(str)]
    for pattern in (("total", "revenue"), ("total revenue",), ("totalrevenue",),
                    ("revenue",), ("net sales",)):
        for r in fin_df.index:
            rl = str(r).lower()
            if all(p in rl for p in pattern):
                return r
    return None


def compute_3y_sales_growth_cagr(ticker_symbol):
    """Computes the 3-Year CAGR for revenue."""
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
        rev_t_minus_3 = revenues.iloc[-4]

        if rev_t_minus_3 <= 0 or pd.isna(rev_t_minus_3) or pd.isna(rev_t):
            return None

        cagr = (rev_t / rev_t_minus_3) ** (1/3.0) - 1.0
        return float(cagr)

    elif len(revenues) >= 2:
        yoy = revenues.pct_change().dropna()
        if yoy.empty:
            return None
        return float(yoy.mean())

    else:
        info = t.info
        return info.get("revenueGrowth")


@app.get("/financials/{ticker}")
def get_financial_data(ticker: str):
    """
    API to return financial metrics for an NSE company.
    User provides only ticker, e.g. 'SUZLON'.
    The API auto-converts to 'SUZLON.NS'.
    """

    ticker_symbol = ticker.upper() + ".NS"

    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info
        computed_sales_growth_3y = compute_3y_sales_growth_cagr(ticker_symbol)

        data = {
            "Ticker": ticker_symbol,
            "DebtEquity": info.get("debtToEquity"),
            "Debt": info.get("totalDebt"),
            "PromoterHolding": info.get("heldPercentInsiders"),
            "SalesGrowth3Y": computed_sales_growth_3y,
            "ProfitGrowth": info.get("earningsGrowth"),
            "EPS": info.get("trailingEps"),
            "DividendYield": info.get("dividendYield"),
            "BookValue": info.get("bookValue"),
            "PB": info.get("priceToBook"),
        }

        return {"status": "success", "data": data}

    except Exception as e:
        return {"status": "error", "message": str(e)}
