# TA-Indicators.py

"""
TA indicators on bar data.
Focus on Momentum and use SPY daily as example
"""


import numpy as np
import pandas as pd
import data_reader
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
from matplotlib.finance import volume_overlay
from matplotlib.dates import num2date
import datetime

from talib import MA_Type
from talib import abstract

filename = "data/SPY.Last.txt"
data = data_reader.read_ninja_txt(filename)
data = data["2016":]

# use log of prices, and prepare both pd.DataFrame and np.array as inputs (for some abstract functions df does not work)
input_df = pd.concat([data.iloc[:, :data.shape[1]-1].apply(np.log), data.iloc[:, data.shape[1]-1]], axis=1)
input_df.columns = map(lambda x: x.lower(), data.columns.tolist())
input_array = dict(zip(input_df.columns, input_df.as_matrix().T))

#####################################
# Overlap

# SMA (for MA's, by default: timeperiod==30, price='close'
log_close_sma = abstract.SMA(input_df, timeperiod=30, price='close')
# EMA
log_close_ema = abstract.EMA(input_df)
# WMA
log_close_wma = abstract.EMA(input_df)
# KAMA - Kaufman Adaptive Moving Average
log_close_kama = abstract.KAMA(input_df)
# MIDPRICE - avg(highest high - lowest low) within the lookback period (default 14)
log_midpr = abstract.MIDPRICE(input_df)
# MIDPOINT - avg(highest close - lowest close) within the lookback period (default 14)
log_midpo = abstract.MIDPOINT(input_df)
# Parabolic Stop and Reverse (SAR) - calculate trailing stop points for long and short positions. SAR = P + A(H-P)
log_sar = abstract.SAR(input_df)
# Bollinger Bands
log_bband_upper, log_bband_middle, log_bband_lower = abstract.BBANDS(input_array, 20, 3, 3)
log_bband_upper = pd.DataFrame(log_bband_upper, index=input_df.index)
log_bband_middle = pd.DataFrame(log_bband_middle, index=input_df.index)
log_bband_lower = pd.DataFrame(log_bband_lower, index=input_df.index)

# plot
fig1 = plt.figure(figsize=(12, 9))
fig1.set_tight_layout(True)
ax1 = plt.subplot(211)
ax1.grid(True)
'''
candles = fin.candlestick_ohlc(ax1,
                               zip(range(len(input_df)), input_df.open, input_df.high, input_df.low, input_df.close),
                               width=1, colorup='#77d879', colordown='#db3f3f')
'''
candles = candlestick2_ohlc(ax1, input_df.open, input_df.high, input_df.low, input_df.close,
                                width=1, colorup='#3c9c73', colordown='#c3145b', alpha=0.9)
#ax1.plot(input_df.close, color="black")
#ax1.plot(log_close_ema, color="green", linewidth=2)
#ax1.plot(log_midpo, color="red", linewidth=1, alpha=0.7)
#ax1.legend(["Close", "EMA", "Mid Point"], loc="upper left", fontsize=10)
#ax1.set_ylabel("Log price")
#ax1.set_title("Historical daily price of SPY (Log)", fontsize=14)
#plt.setp(ax1.get_xticklabels(), fontsize=12)

#xt = ax1.get_xticks()
#new_xticks = [datetime.date.isoformat(num2date(d)) for d in xt]
#ax1.set_xticklabels(new_xticks, rotation=45, horizontalalignment='right')

# shift y-limits of the candlestick plot so that there is space at the bottom for the volume bar chart
pad = 0.5
yl = ax1.get_ylim()
ax1.set_ylim(yl[0]-(yl[1]-yl[0])*pad, yl[1])

xl = ax1.get_xlim()
ax1.set_xlim(xl[0], 3*xl[1])

ax2 = ax1.twinx()
bc = volume_overlay(ax2, input_df.open, input_df.close, input_df.volume, colorup='#0d8cec', colordown='#ff82c7', alpha=0.1, width=1)
ax2.add_collection(bc)
'''
ax3 = plt.subplot(212, sharex=ax1)
ax3.grid()
ax3.plot(input_df.close, color="black")
ax3.plot(log_bband_middle, color="red")
ax3.fill_between(log_bband_middle.index, log_bband_upper.values.flatten(), log_bband_lower.values.flatten(),
                 facecolor="dodgerblue", alpha=0.5)
ax3.legend(["Close", "Middle", "Bollinger (2-SD)"], loc="upper left", fontsize=10)
ax3.set_ylabel("Log price")
ax3.set_xlabel("Time")

plt.show()

#####################################
# Momentum

# ADX
log_mfi = abstract.ADX(input_df)

fig2 = plt.figure(figsize=(12, 9))
ax1 = plt.subplot(211)
ax1.grid()
ax1.plot(input_df.close, color="black")
ax1.set_ylabel("Log price")
ax1.set_title("Historical daily price of SPY (Log)", fontsize=14)

ax2 = plt.subplot(212)
ax2.grid()
ax2.plot(log_mfi, color="green")
#ax1.plot(log_midpo, color="red", linewidth=1, alpha=0.7)
ax2.legend(["MFI", "Mid Point"], loc="upper left", fontsize=10)
'''

plt.show()