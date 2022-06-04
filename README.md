# FYP-scDHA
final year project of John Yeung

This is a python implementation of the scDHA pipeline, with some functions missing for as the analysis is conducted on small dataset only.

Data used in this project can be obtained from https://bioinformatics.cse.unr.edu/software/scDHA/resource/Reproducibility/

The referenced paper can be obtained via https://www.nature.com/articles/s41467-021-21312-2

The main algorithm scDHA is located in scDHA.py, where it will return a list of 9 versions of scDHA compressed input scRNA expression data
To use it, simple import the file and call the function
```
import scDHA
sc = scDHA(data_path)
```

Where denoise.py and encode.py contains the deep learning model used in scDHA.py respectively.

Analysis.py contains the application layer functions like classification (uses knn), clustering (uses functions in clustering.py, simple kmeans/SpectralClustering is used) 
and visualization (only UMAP is implemented)
```
from Analysis import scDHA_class
res = scDHA_class(path_name, scDHA_data, seed=1, retrain=False)

from Analysis import scDHA_vis, scDHA_clus
latent, clusters = scDHA_clus(path_name, scDHA_data, 8)
scDHA_vis(path_name, latent, clusters, 8)
```

There are also functions with the same name that can load the saved latent data according to the path_name.

In scDHA_test.py, it includes the scDHA pipeline with more flexible hyper-parameter options as user input.
```
import scDHA_test
sc = scDHA(path_name, is_plot_denoise, norm='log', denoise_epochs=10, 
          encode_epochs=[10, 20], lr=5e-4, wdecay = [1e-6, 1e-3], seed=None)
```

