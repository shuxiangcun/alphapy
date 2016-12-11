# momentum.py
"""
Usage:
    macd_strategy = MomentumStrategy(log_price)
    enters = macd_strategy.macd_enter(look_back=10)
    strategy_return = macd_strategy.evaluate(enters, hold_period=10)

"""
import pandas as pd
import matplotlib.pyplot as plt

'''TO-DO: cmf, bollinger band, RSI, stochastics
   other momentum features
'''

def sma(price, window=20):
    if len(price) < window:
        raise ValueError("Window longer than price length")

    _sma = pd.Series(0.0, index=price.index)
    for i in range(window-1, len(price)):
        _sma[i] = (price[i-19:i+1].sum()) / window
    return _sma  # [window-1:]


def ema(price, window=10):
    if len(price) < window:
        raise ValueError("Window longer than price length")

    _ema = pd.Series(0.0, index=price.index)
    alpha = 2.0 / (window + 1)
    _ema[window-1] = price[:window].sum() / window  # first ema data point set to sma value
    for i in range(window, len(price)):
        _ema[i] = alpha * price[i] + (1 - alpha) * _ema[i-1]
    return _ema  # [window-1:]


def macd(price, window1=12, window2=26):
    if len(price) < window2:
        raise ValueError("Window longer than price length")

    return ema(price, window1) - ema(price, window2)   # [(window2-window1):]


def cci(price, window=20):
    if len(price) < window:
        raise ValueError("Window longer than price length")

    _sma = pd.Series(0.0, index=price.index)
    _cci = pd.Series(0.0, index=price.index)
    for i in range(window - 1, len(price)):
        _sma[i] = price[i-19:i+1].sum() / window
        _cci[i] = (price[i] - _sma[i]) / (0.015 * price[i-19:i+1].std())

    return _cci  # [window-1:]


def obv(price, volume, window=None):
    if len(price) != len(volume):
        raise Exception("Length of price and volume must equal")

    _sign = np.sign(price.diff())
    _sign = _sign.replace(to_replace=0, method='ffill')  # propagate last valid observation forward to next valid
    _obv = (volume * _sign).cumsum()
    if window is None:
        return _obv  # [1:]

    else:
        if len(price) < window:
            raise ValueError("Window longer than price length")
        _obv2 = pd.Series(0.0, index=volume.index)
        _obv2[window] = _obv[window]
        for i in range(window + 1, len(volume)):
            _obv2[i] = _obv[i] - _obv[i - window]

        return _obv2  # [window:]


#daily_sharpe = lambda x: x.mean()/x.std()

class MomentumStrategy:
    def __init__(self, price, scale="log"):
        self.price = price
        self.scale = scale

    def macd_enter(self, look_back):
        ind_macd = macd(self.price)

        enters = pd.Series(0.0, index=self.price.index)
        for i in range(look_back, len(self.price)):
            if ind_macd[i] > 0 and all(ind_macd[j] <= 0 for j in range(i - look_back, look_back)):
                enters[i] = 1

        return enters

    def evaluate(self, enters, hold_period):
        exits = enters.shift(hold_period).fillna(value=0)

        if self.scale == "log":
            step_period_return = self.price.diff(periods=hold_period)
            holding_period_return = step_period_return * exits
            strategy_return = holding_period_return.cumsum()

        elif self.scale == "level":
            step_period_return = self.price.pct_change(periods=hold_period)
            holding_period_return = step_period_return * exits
            strategy_return = (holding_period_return + 1).cumprod() - 1

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(strategy_return)
        plt.xlabel("Tick")
        plt.ylabel("Cumulative return")
        plt.title("MACD Strategy Cumulative Return")
        fig.savefig("result/momentum/macd_strategy_return.png")
        plt.close(fig)

        return strategy_return
