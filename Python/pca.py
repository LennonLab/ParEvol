from __future__ import division
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt

mydir = os.path.expanduser("~/GitHub/ParEvol/")


def hellinger_transform():
    df = pd.read_csv(mydir + 'data/ltee/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)
    return(df.div(df.sum(axis=1), axis=0).applymap(np.sqrt))


def get_pca():
    X = hellinger_transform()
    pca = PCA()
    X_pca = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)



    fig = plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)

    plt.xlabel('pca1', fontsize = 14)
    plt.ylabel('pca2', fontsize = 14)

    fig.tight_layout()
    fig_name = mydir + 'figs/pca.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()

    #plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    #plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')




#hellinger_transform()
get_pca()
