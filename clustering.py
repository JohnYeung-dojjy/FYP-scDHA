import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torch.optim as optim                          # optimization

import os
from tqdm import tqdm                                # for progress bar
from sklearn.cluster import KMeans, SpectralClustering


def clus(data: np.ndarray, k=None, nmax=10):
    # get no of rows in data
    nrow = data.shape[0]
    print(data.shape)
    if k==None:
        print("finding best k...")
        k = ncluster_par(data, nmax=nmax)
        print(f'best k is {k}')
    
    if nrow < 1000 and k < 5:
        k = KMeans(n_clusters=k, n_init=1000, max_iter=1000).fit(data)
        # return the labels (equal to k$clusters in original code)
        return k.labels_
    else:
        while True:
            try:
                print(f"running SC on k={k}...")
                kknn = SpectralClustering(k, eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=7,  
                                                        assign_labels='kmeans', n_jobs=-1, random_state=0)  # n_jobs = -1 means use all processors                              
                kknn.fit(data)                                                                         
                print(kknn)
                break
            
            except Exception as e: # if input k not work
                print(e)
                k += 1 # try different n_clusters
    # print(f'kknn: {kknn}, k: {k}')    
    return kknn.labels_

# def clus_big(data, k=None, n=2000, nmax=10):
#     tmp_cluster = clus(np.random.shuffle(data)[:n], k, nmax=nmax) # do clustering with n randomly chosen samples
    

def ncluster_par(data, nmax=10):
    """predict number of clusters of the data set"""
    ####### no need to use, as sklearn handles different seeds by default
    result=[]
    for j in range(10):
        # np.random.seed(j)
        # idx = np.random.choice(np.arange(1, data.shape[0]), min(500, data.shape[0]))
        # sub_data = np.Generator.choice(data, min(500, data.shape[0]))
        # print(sub_data.shape)
        np.random.shuffle(data)
        sub_data = data[:min(500, data.shape[0])]
        to_test = np.zeros((nmax, 3))
        e = 2
        sub_data_mean = sub_data.mean(axis=0) # axis=0 mean sample mean
        TSS = np.square(sub_data-sub_data_mean).sum() # total sum of square (sparcity of data)
        for k in range(2, nmax):
            # SS_between, SS_total, SS_within = spec_clust(data[idx], i, nn=7)
            try:
                kknn = SpectralClustering(k,  affinity='nearest_neighbors', n_neighbors=7, n_jobs=-1,  # eigen_solver  is default arpack
                                            random_state=0).fit(sub_data)                                # assign_labels is default Kmeans
                                                                                                    # n_jobs = -1 means use all processors
                WSS=0   # total within sum of squares
                for cell_class in range(k):
                    # print(k, end=' ')
                    mean = sub_data[kknn.labels_ == cell_class].mean(0) # axis=0 mean sample mean
                    WSS += np.square(sub_data[kknn.labels_ == cell_class]-mean).sum()
                BSS = TSS - WSS # between sum of square
                to_test[k-1, 0] = BSS/TSS # (between sum of squares)/(total sum of square)              # index 1
                to_test[k-1, 1] = WSS     # (total within sum of square)
                
            except Exception as ex:
                print(ex)
                e = k-1 + 2
            # print()        
        to_test[e:nmax, 2] = (to_test[e:nmax, 1] - to_test[e-1:nmax-1, 1])/to_test[e-1:nmax-1, 1] # index 2
        # print(to_test, '\n')
        result.append([to_test[:, 0].argmax(), to_test[:, 2].argmax()]) # indices of max argument, mean of them is the predicted n_cluster
    result = np.array(result)
    # print(result)
    return np.floor(result.mean()+0.5).astype(np.int)+1 # original R code is return floor(mean(result) + 0.5)
                                                   # add 1 as python array index starts with 0

