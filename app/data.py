import os
from polygon import StocksClient
import pandas as pd
from datetime import datetime, timedelta

def get_client():
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Set POLYGON_API_KEY in .env")
    return StocksClient(api_key)

def fetch_prices(tickers=["SPY", "AGG"], days=756):  # ~3 years
    client = get_client()
    end = datetime.now()
    start = end - timedelta(days=days + 100)

    dfs = []
    for ticker in tickers:
        response = client.get_aggregate_bars(
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            multiplier=1,
            timespan="day"
        )

        results = response.get("results", []) if isinstance(response, dict) else response
        if not results:
            raise ValueError(f"No data returned for {ticker}")

        records = [{"timestamp": bar["t"], "close": bar["c"]} for bar in results]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')['close'].rename(ticker)
        dfs.append(df)

    prices = pd.concat(dfs, axis=1).dropna()
    return prices.tail(days)
