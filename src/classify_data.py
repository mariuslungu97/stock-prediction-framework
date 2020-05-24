"""
============IMPORTS============
"""
import pandas as pd


class DataClassifier:
    """
    Classifies financial data by labeling each point as a buy(2), hold(0), or sell(1), based on the following params:
        1. target_col
        2. t_in_future
    The classification is performed using dynamic thresholds defined by volatility.
    The classification uses the triple barrier method, defined as follows:
        1. by hitting the upper barrier first (threshold*multiplier), a data point is defined as a buy
        2. by hitting the lower barrier first (threshold*multiplier), a data point is defined as a sell
        3. by hitting the right barrier first (target_col value at horizon point), a data point is defined as a hold
    The triple barrier method mimics a stop-loss/take-profit strategy used in technical analysis.
    The classifier allows for long/short bets.
    """
    def __init__(self):
        pass

    def classify(self, data, target_col, t_in_future, delta=None, span=None):
        """
        :param data: MANDATORY
        :param target_col: MANDATORY
        :param t_in_future: MANDATORY
        :param delta: OPTIONAL; default value: 1;
        :param span: OPTIONAL; default value: 30
        :return: DataFrame with 'label' column
        """
        target_col = target_col.lower()

        # delta and t_in_future specify the differences in number of days
        if delta is None:
            delta = pd.Timedelta(days=1)
        else:
            delta = pd.Timedelta(days=delta)

        if span is None:
            span = 30

        t_in_future = pd.Timedelta(days=t_in_future)

        # calculate volatility, horizon points
        events = pd.DataFrame()
        events = events.assign(threshold=self._get_volatility(data[target_col], delta, span)).dropna()
        events = events.assign(t1=self._get_horizons(data[target_col], t_in_future=t_in_future)).dropna()
        events = events.assign(side=pd.Series(1., events.index))  # only long bets, can be set as -1 for short bets also

        touches = self._get_touches(data[target_col], events)
        touches = self._get_labels(touches)

        return touches.label

    def _get_volatility(self, target_col, delta, span):
        """
        @:param target_col - column whose values are being compared at different times, as pd Series
        @:param delta - the difference in time used to calculate return rate, as pd datetime
        Formula:
            vol = sd([target_col[t] / target_col[t-delta] - 1 for t in target_col])
        Used to compute dynamic thresholds.
        """
        df0 = target_col.index.searchsorted(target_col.index - delta)
        df0 = df0[df0 > 0]
        df0 = pd.Series(target_col.index[df0 - 1], index=target_col.index[target_col.shape[0] - df0.shape[0]:])
        df0 = target_col.loc[df0.index] / target_col.loc[df0.values].values - 1
        df0 = df0.ewm(span=span).std()
        return df0

    def _get_horizons(self, target_col, t_in_future):
        """
        @:param target_col - column whose values are being compared at different times, as pd Series
        @:param t_in_future - the difference in time between target_col[t] and its horizon, as pd datetime
        Used to map all data points to their future horizons, based on a time in future
        """
        horizons = target_col.index.searchsorted(target_col.index + t_in_future)
        horizons = horizons[horizons < target_col.shape[0]]
        horizons = target_col.index[horizons]
        horizons = pd.Series(horizons, index=target_col.index[:horizons.shape[0]])
        return horizons

    def _get_touches(self, target_col, events):
        """
        Sets upper and lower barriers on which the labeling will be decided.
        events is a DataFrame containing:
            t1: timestamp of the next horizon, calculated using _get_horizons
            threshold: unit height of lower and upper barriers, calculated using _get_volatility
            side: the side (long or short) for each bet
        """
        thresh_upper, thresh_lower = 1, 1
        out = events[["t1"]].copy(deep=True)

        thresh_upper = thresh_upper * events["threshold"]
        thresh_lower = - thresh_lower * events["threshold"]

        for t, t1 in events["t1"].iteritems():
            res = target_col[t:t1]
            res = (res / target_col[t] - 1) * events.loc[t, "side"]

            out.loc[t, "stop_loss"] = res[res < thresh_lower[t]].index.min()
            out.loc[t, "take_profit"] = res[res > thresh_upper[t]].index.min()

        return out

    def _get_labels(self, touches):
        """
        Based on the calculated touches, return the corresponding data labels
        """
        out = touches.copy(deep=True)
        first_touch = touches[["stop_loss", "take_profit"]].min(axis=1)

        for t, t1 in first_touch.iteritems():
            if pd.isnull(t1):
                out.loc[t, "label"] = 0
            elif t1 == touches.loc[t, "stop_loss"]:
                out.loc[t, "label"] = 1
            else:
                out.loc[t, "label"] = 2
        return out
