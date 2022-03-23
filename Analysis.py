"""
#Generate clustering result, the input matrix has rows as samples and columns as genes
#'   result <- scDHA(data, ncores = 2, seed = 1)
#'   #Generate 2D representation, the input is the output from scDHA function
#'   result <- scDHA.vis(result, ncores = 2, seed = 1)
#'   #Plot the representation of the dataset, different colors represent different cell types
#'   plot(result$pred, col=factor(label), xlab = "scDHA1", ylab = "scDHA2")
"""

import umap # Uniform Manifold Approximation and Projection for Dimension Reduction
import numpy as np


# def scDHA_vis(latent: np.ndarray,  method="UMAP"):
#     if method == "UMAP":
#         reducer = umap.UMAP()
#         return reducer.fit_transform(latent)
    
#     elif method == "scDHA":
#         return scDHA_vis_old(latent)
    
#     else:
#         exit('\'method\' should be \'UMAP\' or \'scDHA\'')
        
        
# def scDHA_vis_old(latent: np.ndarray):
#     if latent.shape[0] > 5e4:

from scDHA import scDHA
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import correlation
from scipy.stats import mode
import random
import pandas as pd

from clustering import clus
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

import umap
import os

def scDHA_class(path_name, n_jobs = -1, seed = None, retrain = False):
    """This method does classification on the dataset

    Args:
        path_name (string): the name of the dataset
        n_jobs (int, optional): number of cpu cores used in knn. Defaults to -1 (use all cores).
        seed ([int], optional): random number seed, so result can be reproduced. Defaults to None.

    Returns:
        numpy array: list of 9 classification results
    """
    latent = scDHA(path_name, False, vae_choice='paper', retrain=retrain, seed=1)
    y_df = pd.read_csv(f'rds_csv_data/{path_name}_labels.csv')['x']
    y = np.array(y_df)

    # 75% train, 25% test
    spilt = int(len(y)*0.75)
    
    np.random.seed(seed)
    
    # choose random indices
    # 2000 times faster than the list comprehension and marginally faster than np.delete
    # referenced to https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list
    train_idx = np.random.choice(len(y), spilt, replace = False)
    test_idx = np.ones(len(y), dtype=bool)
    test_idx[train_idx] = False

    y_train, y_test = y[train_idx], y[test_idx]
    
    results = []
    for x in latent:
        random.seed(seed)
        x_train, x_test = x[train_idx], x[test_idx]
        knn = KNeighborsClassifier(n_neighbors=10, metric=correlation, n_jobs=-n_jobs)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        results.append(y_pred)

    result = mode(results).mode
    
    print('\n####################')
    print(f'    accuracy: {(result==y_test).mean():.2f}')
    print('####################')
    return result


def scDHA_clus(path_name, n_clusters):
    data = np.load(f'latent/{path_name}.npy')

    y_df = pd.read_csv(f'rds_csv_data/{path_name}_labels.csv')['x']
    y = np.array(y_df)

    best_ari = 0
    for latent in data:
        cluster = clus(latent, k=n_clusters)
        
        ari = adjusted_rand_score(y, cluster)
        if ari > best_ari:
            best_latent = latent
            best_ari = ari
            best_cluster = cluster
    np.save(f'clustering/{path_name}_latent.npy', best_latent)
    np.save(f'clustering/{path_name}_label.npy' , best_cluster)

    print(f'ari: {best_ari}')
    return best_latent, best_cluster

def scDHA_vis(path_name, n_clusters, method='umap'):
    if os.path.isfile(f'clustering/{path_name}_latent.npy'):
        best_latent = np.load(f'clustering/{path_name}_latent.npy')
        best_cluster = np.load(f'clustering/{path_name}_label.npy')
    else:
        best_latent, best_cluster = scDHA_clus(path_name, n_clusters)
    
    if method != 'umap':
        print('method can only be umap because I only implemented this, using umap instead')
            
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(best_latent)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=best_cluster, cmap='Spectral', s=5)
    plt.title(f'UMAP projection of the {path_name} latent variables', fontsize=12)
    plt.show()