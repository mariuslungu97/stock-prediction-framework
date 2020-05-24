"""
============IMPORTS============
"""
import pandas as pd
from os import path
import glob
from talib import abstract


class DataLoader:
    """
        Provides abstractions for:
            1. loads tickers' data into memory (disk or api)
            2. standardises datasets into expected format
            3. calculates technical indicators
            3. merges tickers' data into single DataFrame
    """
    def __init__(self, tickers, data_type, tech_indicators):
        self.tickers = tickers
        self.data_type = data_type
        self.tech_indicators = tech_indicators
        self.main_df = pd.DataFrame()

    def load_data(self, use_adj=None):
        """
        Loads all files with the requested ticker symbols in the beginning of their file names:
            aapl.us.txt, tsla_stock.csv
        The files are expected to contain comma-separated values.
        If there are multiple files with same ticker symbol, it will only load the first one in the dir.
        @:param use_adj - use the adjusted prices to accurately reflect the stock value after accounting for corporate actions
        :return: DataFrame containing the merged columns of all required tickers
        """
        if use_adj is None:
            use_adj = True

        main_df = pd.DataFrame()

        for ticker in self.tickers:
            ticker = ticker.lower()

            file_path = path.join("data", f"{self.data_type}_data", f"{ticker}*.*")

            # first file with ticker name in dir
            ticker_files = glob.glob(file_path)[0] if len(glob.glob(file_path)) > 1 else glob.glob(file_path)
            ticker_files = [ticker_files] if isinstance(ticker_files, str) else ticker_files

            for filename in ticker_files:
                temp_df = pd.read_csv(filename, index_col=0)
                temp_df = self._format_data(temp_df, use_adj)
                # calculate technical indicators, if required
                if self.tech_indicators is not None:
                    t_data = DataLoader.calculate_tech_indicators(temp_df, self.tech_indicators)
                    temp_df = pd.concat([temp_df, t_data], axis=1)
                # rename columns to identify by ticker name
                temp_df = temp_df.rename(columns={col: f"{ticker}_{col}" for col in temp_df.columns})
                # merge data sets
                if len(main_df) == 0:
                    main_df = temp_df
                else:
                    main_df = main_df.join(temp_df, how="outer")

        self.main_df = main_df
        # return merged dataset
        return self.main_df

    def _format_data(self, data, use_adj):
        # column headers to lowercase
        data = data.rename(
            columns={col: col.lower() for col in data.columns}
        )
        # format datetime index
        data.index = pd.to_datetime(pd.to_datetime(data.index).strftime("%Y-%m-%d"))
        data = data.sort_index(ascending=True)

        # calculate and use adjusted ohlc columns if required; else use original ohlc cols
        if use_adj and self.data_type == "api":
            data_ohlcv_adj = data["adjusted close"].resample("1D").ohlc()
            data_ohlcv_adj["volume"] = data["volume"]
            data = data_ohlcv_adj.loc[data.index, :]
        else:
            data = data[["open", "high", "low", "close", "volume"]]

        return data

    @staticmethod
    def calculate_tech_indicators(data, tech_indicators):
        """
        Calculates technical indicators, provided data as Dataframe and list of technical indicators.
        List of supported technical indicators: http://mrjbq7.github.io/ta-lib/
        @:return Dataframe with tech indicators
        """
        tech_indicators = [tech_indicators] if not isinstance(tech_indicators, list) else tech_indicators

        results = []
        for tech_indicator in tech_indicators:
            tech_indicator = tech_indicator.lower()
            t_func = abstract.Function(tech_indicator)
            t_data = t_func(data)
            # Series to Data Frame
            if isinstance(t_data, pd.Series):
                t_data = pd.DataFrame({tech_indicator: t_data.values}, index=t_data.index)
            results.append(t_data)

        results = pd.concat(results, axis=1)
        return results


