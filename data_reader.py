# data_reader.py
"""
Created on 10/15/2016 by LT.

Wrapped data-readin applications.

1. read_ninja_csv
2. read_ninja_multi
"""

import pandas as pd
import csv

def read_csv(file_name, dialect='rb', delimiter=",", has_index=False, has_column=True):
    """Read in csv.

        Args:
            file_name
            dialect
            delimiter
            index
            column

        Returns:
            dataset with index and columns set up.
    """

    data = []
    columns = []
    file = open(file_name, dialect)
    try:
        reader = csv.reader(file, delimiter=delimiter)
        # return an iterator with each iteration being a row of the data file
        for idx, row in enumerate(reader):
            if (has_column) & (idx == 0):
                columns.append(row)
                continue
            data.append(row)
    finally:
        file.close()

    data = pd.DataFrame(data)

    if has_index:
        data_index = pd.to_datetime(data[0])
        data.index = data_index  # assign index
        data = data.iloc[:, (-data.shape[1] + 1):]
    
    if has_column:
        data.columns = columns[0]

    for i in range(data.shape[1]):
        data.iloc[:, i] = data.iloc[:, i].apply(float)  # convert str to float

    return data

def read_ninja_txt(file_name, dialect='rb', delimiter=";"):
    """Read in .txt files (OHLCV) downloaded from Ninja.
    
    Args:
        file_name
        dialect
        delimiter
    
    Returns:
        dataset with index and columns set up.

    """

    # with open(file_name) as f:
    #     data = f.readlines()
    # f.close()
    
    data = []
    file = open(file_name, dialect)
    try:
        reader = csv.reader(file, delimiter=delimiter)
        # return an iterator with each iteration being a row of the data file      
        for idx, row in enumerate(reader):
            data.append(row)
    finally:
        file.close()
    
    data = pd.DataFrame(data)
    
    data_index = pd.to_datetime(data[0])
    data.index = data_index  # assign index
    data = data.iloc[:, (-data.shape[1]+1):]
    data.columns = ["Open", "High", "Low", "Close", "Volume"]  # assign columns
    
    for i in range(data.shape[1]):
        data.iloc[:, i] = data.iloc[:, i].apply(float)  # convert str to float
    
    return data


def read_ninja_multi(file_names, columns, field="Close"):
    """Read in multiple .txt files (OHLCV) downloaded from Ninja and return
    a data frame of one specified variable, by default, "Close".

    Args:
        file_names
        columns
        field

    Returns:
        data frame with all tickers specified column.
    """

    data = pd.DataFrame(index=read_ninja_txt(file_names[0]).index)
    for file in file_names:
        df = read_ninja_txt(file)
        data = pd.concat([data, df[field]], axis=1)
    data.columns = columns

    return data


def read_usequity_tickfile(file_name, dialect='rb', delimiter=","):
    """Read in the .csv file from the tick data package (US_Equity), for play with tick data.

    Args:
        file_name
        dialect
        delimiter

    Returns:
        dataset with columns set up.

    """

    # with open(file_name) as f:
    #     data = f.readlines()
    # f.close()

    data = []
    file = open(file_name, dialect)
    try:
        reader = csv.reader(file, delimiter=delimiter)
        # return an iterator with each iteration being a row of the data file
        for idx, row in enumerate(reader):
            data.append(row)
    finally:
        file.close()

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

"""
def dead_wrds_csv(file_name, )

    file_name = "data/wrds/2f288223e80b0f04.csv"
    dialect = "rb"
    delimiter = ','
    
    data = []
    f = open(file_name, dialect)
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
"""

def read_tickers(file_name, dialect='rb', delimiter=","):
    """Read tickers used to download data.

    Args:
        file_name
        dialect
        delimiter

    Returns:
        dataset with columns set up.

    """
    tickers = []
    file = open(file_name, dialect)
    try:
        reader = csv.reader(file, delimiter=delimiter)
        # return an iterator with each iteration being a row of the data file
        for idx, row in enumerate(reader):
            tickers.append(row)
    finally:
        file.close()
    tickers = [ticker[0] for ticker in tickers]
    return tickers
