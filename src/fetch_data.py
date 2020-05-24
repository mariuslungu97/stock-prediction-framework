"""
============IMPORTS============
"""
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from os import path
from os import makedirs
import sys
import re
import traceback


class DataFetcher:
    """
        Provides abstractions for:
            1. fetch data from AlphaVantage API
            2. saves data to ../data/api_data

        Momentarily only compatible with the AlphaVantage API.
    """
    def __init__(self, api_key):
        self._api_key = api_key
        self.ts = TimeSeries(key=self._api_key, output_format="pandas")

    def fetch_data(self, tickers, time_window):

        # format tickers as list
        tickers = [tickers] if not isinstance(tickers, list) else tickers

        try:
            if len(tickers) <= 5:

                for ticker in tickers:
                    ticker = ticker.lower()

                    if time_window == "daily":
                        data, meta_data = self.ts.get_daily_adjusted(symbol=ticker, outputsize="full")
                    elif time_window == "weekly":
                        data, meta_data = self.ts.get_weekly_adjusted(symbol=ticker, outputsize="full")
                    elif time_window == "monthly":
                        data, meta_data = self.ts.get_monthly_adjusted(symbol=ticker, outputsize="full")
                    else:
                        # time_window parameter not supported, raise exception
                        raise TypeError(f"The following time_window: '{time_window}' is not supported momentarily")

                    # format and convert index to datetime
                    data.index = pd.to_datetime(data.index.strftime("%Y-%m-%d"))
                    data = data.sort_index(ascending=True)

                    # remove numbers from column names
                    data = data.rename(columns={col: re.sub(r"([0-9]\. )", "", col) for col in data.columns})
                    data = data.rename(columns={col: col.lower() for col in data.columns})

                    # create api_data directory
                    if not path.exists(path.join("..", "data", "api_data")):
                        makedirs(path.join("..", "data", "api_data"))

                    # save data to ./data/api_data
                    save_path = path.join(
                        path.dirname(path.abspath(__file__)),
                        "..",
                        "data",
                        "api_data",
                        f"{ticker}_{time_window}.csv"
                    )
                    data.to_csv(save_path)
                return True
            else:
                # no more than 5 API calls (tickers) per minute, raise exception
                raise ValueError("No more than 5 API calls/minute supported! Please provide a ticker list <= 5")
        except (ValueError, TypeError) as e:
            traceback.print_exc()
            print(e)
            sys.exit(0)
