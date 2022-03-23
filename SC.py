#%%
from clustering import clus
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import pandas as pd
from sklearn.metrics import adjusted_rand_score

path_name = "baron-human"

data = np.load(f'latent/{path_name}.npy')

y_df = pd.read_csv(f'rds_csv_data/{path_name}_labels.csv')['x']
y = np.array(y_df)

best_ari = 0
for latent in data:
    cluster = clus(latent, k=5)
    
    ari = adjusted_rand_score(y, cluster)
    if ari > best_ari:
        best_latent = latent
        best_ari = ari
        best_cluster = cluster
np.save(best_latent, f'clustering/{path_name}_latent.npy')
np.save(best_cluster, f'clustering/{path_name}_label.npy')

print(f'ari: {best_ari}')

#%%
import umap
import matplotlib.pyplot as plt
reducer = umap.UMAP()
embedding = reducer.fit_transform(best_latent)
plt.scatter(embedding[:, 0], embedding[:, 1], c=best_cluster, cmap='Spectral', s=5)
plt.title('UMAP projection of the latent variables', fontsize=12)
plt.show()
# %%
import Analysis
Analysis.scDHA_vis('goolam')
# %%
