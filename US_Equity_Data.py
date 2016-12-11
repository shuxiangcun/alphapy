# Momentum.py
"""Analyze tick data to extract momentum patterns
"""

import numpy as np
import pandas as pd
import data_reader
import matplotlib.pyplot as plt


def main():
    """
    data = []
    f = open(filename)
    try:
        reader = csv.reader(f, delimiter=',')
        # return an iterator with each iteration being a row of the data file
        for idx, row in enumerate(reader):
            data.append(row)
    finally:
        f.close()

    # pre-processing

    data = pd.DataFrame(data)
    data = data.iloc[:, -7:-1]
    data.columns = ["bid_price", "bid_size", "ask_price", "ask_size", "last_transact_price", "last_transact_size"]
    for i in range(data.shape[1]):
        data.iloc[:, i] = data.iloc[:, i].apply(float)  # convert str to float
    """

    filename = "data/US_Equity/SPY/20160302/SPY.csv"
    data = data_reader.read_usequity_tickfile(filename)
    data = data[-2000:]  # use first 20k ticks

    # calculate weighted mid mkt
    data["weighted_mid_price"] = (data["bid_price"] * data["ask_size"] + data["ask_price"] * data["bid_size"]) / \
                                 (data["bid_size"] + data["ask_size"])
    # calculate spread
    data["spread"] = data["ask_price"] - data["bid_price"]

    # use wmm to define continuous up/downs in tick price (3, 4, 5 ups or downs in a row)
    wmm_up_t = data["weighted_mid_price"].diff() > 0  # uptick
    wmm_down_t = data["weighted_mid_price"].diff() < 0  # uptick

    wmm_up_3t = pd.Series(0.0, index=data.index)
    wmm_up_4t = pd.Series(0.0, index=data.index)
    wmm_up_5t = pd.Series(0.0, index=data.index)
    wmm_down_3t = pd.Series(0.0, index=data.index)
    wmm_down_4t = pd.Series(0.0, index=data.index)
    wmm_down_5t = pd.Series(0.0, index=data.index)
    
    for i in range(2, len(data)):
        if all(wmm_up_t[j] for j in range(i - 2, i + 1)):
            wmm_up_3t[i] = 1
    for i in range(3, len(data)):
        if all(wmm_up_t[j] for j in range(i - 3, i + 1)):
            wmm_up_4t[i] = 0.75
    for i in range(4, len(data)):
        if all(wmm_up_t[j] for j in range(i - 4, i + 1)):
            wmm_up_5t[i] = 0.5
    for i in range(2, len(data)):
        if all(wmm_down_t[j] for j in range(i - 2, i + 1)):
            wmm_down_3t[i] = -1
    for i in range(3, len(data)):
        if all(wmm_down_t[j] for j in range(i - 3, i + 1)):
            wmm_down_4t[i] = -0.75
    for i in range(4, len(data)):
        if all(wmm_down_t[j] for j in range(i - 4, i + 1)):
            wmm_down_5t[i] = -0.5

    # compare last transact price to bid/ask/wmm and define momentum
    transact_up_t = data["last_transact_price"] >= data["ask_price"]  # last transact at or above ask
    transact_down_t = data["last_transact_price"] <= data["bid_price"]  # last transact at or below bid

    transact_up_10t = pd.Series(0.0, index=data.index)
    transact_up_20t = pd.Series(0.0, index=data.index)
    transact_up_30t = pd.Series(0.0, index=data.index)
    transact_down_10t = pd.Series(0.0, index=data.index)
    transact_down_20t = pd.Series(0.0, index=data.index)
    transact_down_30t = pd.Series(0.0, index=data.index)

    for i in range(9, len(data)):
        if all(transact_up_t[j] for j in range(i - 9, i + 1)):
            transact_up_10t[i] = 1
    for i in range(19, len(data)):
        if all(transact_up_t[j] for j in range(i - 19, i + 1)):
            transact_up_20t[i] = 0.75
    for i in range(29, len(data)):
        if all(transact_up_t[j] for j in range(i - 29, i + 1)):
            transact_up_30t[i] = 0.5
    for i in range(9, len(data)):
        if all(transact_down_t[j] for j in range(i - 9, i + 1)):
            transact_down_10t[i] = -1
    for i in range(19, len(data)):
        if all(transact_down_t[j] for j in range(i - 19, i + 1)):
            transact_down_20t[i] = -0.75
    for i in range(29, len(data)):
        if all(transact_down_t[j] for j in range(i - 29, i + 1)):
            transact_down_30t[i] = -0.5

    #######################################
    # plot

    # 1. plot wmm along with bid, ask, and last transacted price     
    fig1 = plt.figure(figsize=(12, 9))
    fig1.set_tight_layout(True)
    ax1 = plt.subplot(211)
    ax1.grid()
    ax1.plot(data[["bid_price", "ask_price", "weighted_mid_price"]])
    ax1.legend(["bid", "ask", "wmm"], loc="upper left", fontsize=10)
    ax1.set_ylabel("($)")
    ax1.set_title("Tick Prices (SPY)", fontsize=14)
    
    plt.setp(ax1.get_xticklabels(), fontsize=12)
    
    # plot spread
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(data["spread"], '.', color="orange")
    ax2.grid()
    ax2.set_xlabel("Time")
    ax2.set_ylabel("($)")
    ax2.legend(["Spread"], loc="upper left", fontsize=10)

    ax3 = ax2.twinx()
    ax3.plot(data["last_transact_size"], color="dodgerblue")
    ax3.set_ylabel("volume")
    ax3.legend(["Volume"], loc="upper center", fontsize=10)

    fig1.savefig("result/us_equity_data/1.tick_price_and_spreads.png")

    # 2. plot prices along with defined tick ups/downs  
    fig2 = plt.figure(figsize=(12, 9))
    fig2.set_tight_layout(True)
    ax1 = plt.subplot(111)
    ax1.grid()
    ax1.plot(data[["bid_price", "ask_price", "weighted_mid_price", "last_transact_price"]])
    ax1.legend(["bid", "ask", "wmm", "last_transact"], loc="upper left", fontsize=8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("($)")

    ax2 = ax1.twinx()
    ax2.plot(wmm_up_3t.index[wmm_up_3t == 1], wmm_up_3t[wmm_up_3t == 1],
             '+', color="lightgreen")
    ax2.plot(wmm_up_4t.index[wmm_up_4t == 0.75], wmm_up_4t[wmm_up_4t == 0.75],
             '+', color="green")
    ax2.plot(wmm_up_5t.index[wmm_up_5t == 0.5], wmm_up_5t[wmm_up_5t == 0.5],
             '+', color="darkslategray")
    ax2.plot(wmm_down_5t.index[wmm_down_5t == -0.5], wmm_down_5t[wmm_down_5t == -0.5],
             'x', color="midnightblue")
    ax2.plot(wmm_down_4t.index[wmm_down_4t == -0.75], wmm_down_4t[wmm_down_4t == -0.75],
             'x', color="blue")
    ax2.plot(wmm_down_3t.index[wmm_down_3t == -1], wmm_down_3t[wmm_down_3t == -1],
             'x', color="lightblue")
    ax2.grid()
    ax2.set_ylim(-1.5, 1.5)
    #ax2.set_ylabel("Indicator")
    ax2.legend(["3-tick up", "4-tick up", "5-tick up", "5-tick down", "4-tick down", "3-tick down"],
               loc="upper center", ncol=2, fontsize=8)
    ax2.set_title("Tick Price Move and Momentum Indicators (SPY)", fontsize=14)

    fig2.savefig("result/us_equity_data/2.tick_price_and_wmm_up_downs.png")

    # 3. plot prices along with defined transact price ups/downs
    fig3 = plt.figure(figsize=(12, 9))
    fig3.set_tight_layout(True)
    ax1 = plt.subplot(111)
    ax1.grid()
    ax1.plot(data[["bid_price", "ask_price", "weighted_mid_price", "last_transact_price"]])
    ax1.legend(["bid", "ask", "wmm", "last_transt"], loc="upper left", fontsize=8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("($)")

    ax2 = ax1.twinx()
    ax2.plot(transact_up_10t.index[transact_up_10t == 1], transact_up_10t[transact_up_10t == 1],
             '+', color="lightgreen")
    ax2.plot(transact_up_20t.index[transact_up_20t == 0.75], transact_up_20t[transact_up_20t == 0.75],
             '+', color="green")
    ax2.plot(transact_up_30t.index[transact_up_30t == 0.5], transact_up_30t[transact_up_30t == 0.5],
             '+', color="darkslategray")
    ax2.plot(transact_down_30t.index[transact_down_30t == -0.5], transact_down_30t[transact_down_30t == -0.5],
             'x', color="midnightblue")
    ax2.plot(transact_down_20t.index[transact_down_20t == -0.75], transact_down_20t[transact_down_20t == -0.75],
             'x', color="blue")
    ax2.plot(transact_down_10t.index[transact_down_10t == -1], transact_down_10t[transact_down_10t == -1],
             'x', color="lightblue")
    ax2.grid()
    ax2.set_ylim(-1.5, 1.5)
    #ax2.set_ylabel("Indicator")
    ax2.legend(["10-tick trade@ask", "20-tick trade@ask", "30-tick trade@ask",
                "30-tick trade@bid", "20-tick trade@bid", "10-tick trade@bid"],
               loc="upper center", ncol=2, fontsize=8)
    ax2.set_title("Tick Price Move and Transacted Price Pattern (SPY)", fontsize=14)

    fig3.savefig("result/us_equity_data/3.tick_price_and_transacted_price_pattern.png")
    
    plt.close('all')
    
if __name__ == "__main__":
    main()
