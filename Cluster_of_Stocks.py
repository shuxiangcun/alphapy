# cluster_of_stocks.py

"""
Use hierarchical clustering to cluster stocks/ETFS and evaluate results.

Selected stocks across sectors and expected to see sector clustering.

Steps:
1. Deciding which variables to use as characteristics to check the similarity.
2. Standardizing the variables. This point is very important as variables with 
   large values could contribute more to the distance measure than variables with small values.
3. Stabilizing the criterion to determine similarity or distance between objects.
4. Selecting the criterion for determining which clusters to merge at successive steps, 
   that is, which hierarchical clustering algorithm to use.
5. Setting the number of clusters needed to represent data.

What to do with the clusters? (assume use haracteristics: the return and volatility over the previous six months)
6. Select the cluster which has the maximum performance and the minimum volatility, in order
   to select it we sort the clusters by performance and volatilility and we choose the one which is on the top
   If there are two clusters with the same position, we select the one with higher performance.
   Then we invest in each asset that the cluster is composed of, equally weighted invested.

Created on Wed Nov 09 2016
@author: Linchang
"""

import data_reader
import clustering
import numpy as np


def main():
    #######################
    # 1. Read in data
    filenames = ["data/AXP.Last.txt", "data/AIG.Last.txt", "data/BAC.Last.txt", "data/BRK.B.Last.txt", "data/C.Last.txt",
                 "data/CB.Last.txt", "data/GS.Last.txt", "data/JPM.Last.txt", "data/MET.Last.txt", "data/MS.Last.txt",
                 "data/PNC.Last.txt", "data/USB.Last.txt", "data/WFC.Last.txt", "data/BK.Last.txt", "data/BLK.Last.txt",
                 "data/PRU.Last.txt", "data/COF.Last.txt", "data/SCHW.Last.txt", "data/CME.Last.txt", "data/MMC.Last.txt"]
    columns = ["AXP", "AIG", "BAC", "BRK.B", "C", "CB", "GS", "JPM", "MET", "MS", "PNC", "USB", "WFC", "BK", "BLK", "PRU",
               "COF", "SCHW", "CME", "MMC"]
    data = data_reader.read_ninja_multi(filenames, columns)
    log_returns = data.apply(np.log).diff()
    log_returns = log_returns["2003":]

    cl = clustering.HierarchCluster(log_returns, columns)
    cl.build_up()

    cl.build_to(1)
    cl.draw_dendrogram('result/cluster/clusters.jpg')


if __name__=="__main__":
    main()
