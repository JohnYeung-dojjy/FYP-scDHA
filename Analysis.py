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


def scDHA_vis(latent: np.ndarray,  method="UMAP"):
    if method == "UMAP":
        reducer = umap.UMAP()
        return reducer.fit_transform(latent)
    
    elif method == "scDHA":
        return scDHA_vis_old(latent)
    
    else:
        exit('\'method\' should be \'UMAP\' or \'scDHA\'')
        
        
def scDHA_vis_old(latent: np.ndarray):
    if latent.shape[0] > 5e4:
        