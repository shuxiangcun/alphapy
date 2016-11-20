# data_reader.py
"""
Created on 10/15/2016 lc.

Wrapped data-readin applications.

1. read_ninja_txt
2. read_ninja_multi
"""

import pandas as pd 

def read_ninja_txt(filename, dialect='rb', delimiter=";"):
    '''Read in .txt files (OHLCV) downloaded from Ninja.
    
    Args:
        filename
        delimiter
    
    Returns:
        dataset with index and columns set up.
        

    '''  
    import csv  

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
    data.index = data_index # assign index
    data = data.iloc[:, (-data.shape[1]+1):]
    data.columns = ["Open", "High", "Low", "Close", "Volume"] # assign columns
    
    for i in range(data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i].apply(float) # convert str to float 
    
    return data


def read_ninja_multi(filenames, columns, field="Close"):
    '''Read in multiple .txt files (OHLCV) downloaded from Ninja and return
    a data frame of one specified variable, by default, "Close".

    Args:
        filename
        column

    Returns:
        data frame with all tickers specified column.
    '''
    data = pd.DataFrame(index=read_ninja_txt(filenames[0]).index)
    for file in filenames:
        df = read_ninja_txt(file)
        data = pd.concat([data, df[field]], axis=1)
    data.columns = columns

    return data
