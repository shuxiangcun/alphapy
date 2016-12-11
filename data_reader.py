# data_reader.py
"""
Created on 10/15/2016 by LC

Wrapped data-readin applications.

1. read_ninja_csv
2. read_ninja_multi
"""

import pandas as pd
import csv


def read_ninja_txt(filename, dialect='rb', delimiter=";"):
    """Read in .txt files (OHLCV) downloaded from Ninja.
    
    Args:
        filename
        dialect
        delimiter
    
    Returns:
        dataset with index and columns set up.

    """

    # with open(filename) as f:
    #     data = f.readlines()
    # f.close()
    
    data = []
    f = open(filename, dialect)
    try:
        reader = csv.reader(f, delimiter=delimiter)
        # return an iterator with each iteration being a row of the data file      
        for idx, row in enumerate(reader):
            data.append(row)
    finally:
        f.close()
    
    data = pd.DataFrame(data)
    
    data_index = pd.to_datetime(data[0])
    data.index = data_index  # assign index
    data = data.iloc[:, (-data.shape[1]+1):]
    data.columns = ["Open", "High", "Low", "Close", "Volume"]  # assign columns
    
    for i in range(data.shape[1]):
        data.iloc[:, i] = data.iloc[:, i].apply(float)  # convert str to float
    
    return data


def read_ninja_multi(filenames, columns, field="Close"):
    """Read in multiple .txt files (OHLCV) downloaded from Ninja and return
    a data frame of one specified variable, by default, "Close".

    Args:
        filenames
        columns
        field

    Returns:
        data frame with all tickers specified column.
    """

    data = pd.DataFrame(index=read_ninja_txt(filenames[0]).index)
    for file in filenames:
        df = read_ninja_txt(file)
        data = pd.concat([data, df[field]], axis=1)
    data.columns = columns

    return data


def read_usequity_tickfile(filename, dialect='rb', delimiter=","):
    """Read in the .csv file from the tick data package (US_Equity), for play with tick data.

    Args:
        filename
        dialect
        delimiter

    Returns:
        dataset with columns set up.

    """

    # with open(filename) as f:
    #     data = f.readlines()
    # f.close()

    data = []
    f = open(filename, dialect)
    try:
        reader = csv.reader(f, delimiter=delimiter)
        # return an iterator with each iteration being a row of the data file
        for idx, row in enumerate(reader):
            data.append(row)
    finally:
        f.close()

    data = pd.DataFrame(data)

    data_index = pd.to_datetime(data[1])  # convert to datetime type
    for i in range(len(data_index)):
        data_index[i] = data_index[i].to_datetime().replace(2016, 02, 03)
    data.index = data_index  # assign index

    data = data.iloc[:, -7:-1]
    data.columns = ["bid_price", "bid_size", "ask_price", "ask_size", "last_transact_price", "last_transact_size"]

    for i in range(data.shape[1]):
                data.iloc[:, i] = data.iloc[:, i].apply(float)  # convert str to float

    return data
