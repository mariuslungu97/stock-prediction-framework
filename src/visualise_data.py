"""
============IMPORTS============
"""
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

class DataVisualiser:
    def __init__(self):
        pass

    @staticmethod
    def visualise_df_ohlc(df, ticker):
        # format df before plotting
        df.index.name = "Date"
        df = df.rename(columns={col: col.replace(ticker.lower() + "_", "").capitalize() for col in df.columns})

        # rescale data into 3-month data
        df_monthly = df["Close"].resample("6M").ohlc()
        df_volume = df["Volume"].resample("6M").sum()

        df_monthly = df_monthly.rename(columns={col: col.capitalize() for col in df_monthly.columns})
        df_monthly["Volume"] = df_volume

        mpf.plot(df_monthly,
                 type="candle",
                 volume=True,
                 style="charles",
                 title=f"{ticker.capitalize()} Data",
                 ylabel="OHLC Candles",
                 ylabel_lower="Share Volume")


    @staticmethod
    def visualise_clf(data, target_col):
        target_col = target_col.lower()
        colors = ["black", "red", "green"]  # matches hold, sell, buy index
        markers = [".", "v", "^"]
        for index, row in data.iterrows():
            label = int(row["label"])
            marker = markers[label]
            color = colors[label]
            plt.plot(index, row[target_col], color=color, marker=marker, markersize=2)
        plt.show()

    @staticmethod
    def plot_results(predicted_data, true_data):
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(true_data, label="True Data")
        plt.plot(predicted_data, label="Prediction")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_results_multiple(predicted_data, true_data, prediction_len):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        plt.show()

    @staticmethod
    def visualise_col(target_col, name):
        plt.figure(figsize=(16, 8))
        plt.title(f"{name} History")
        plt.plot(target_col)
        plt.xlabel("Date", fontsize=18)
        plt.ylabel(f"{name}", fontsize=18)
        plt.show()

    @staticmethod
    def visualise_correlation_table(df):
        df_corr = df.corr()
        data = df_corr.values

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
        fig.colorbar(heatmap)
        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        column_labels = df_corr.columns
        row_labels = df_corr.index

        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(row_labels)
        plt.xticks(rotation=90)
        heatmap.set_clim(-1, 1)
        plt.tight_layout()
        plt.show()

