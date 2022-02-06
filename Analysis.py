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

    result = mode(results)
    
    print('\n####################')
    print(f'    accuracy: {(result.mode==y_test).mean():.2f}')
    print('####################')
    return mode(results)