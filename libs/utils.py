import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

## File functions
def save_pickle(object, filePath):
    with open(filePath, 'wb') as file:
        pickle.dump(object, file)


def load_pickle(filePath):
    with open(filePath, 'rb') as file:
        pickled_data = pickle.load(file)
    return pickled_data


def save_npy(array, filePath):
    with open(filePath, 'wb') as file:
        np.save(file, array, allow_pickle=True)


def load_npy(filePath):
    with open(filePath, 'rb') as file:
        array = np.load(file, allow_pickle=True)
    return array

## Dimensionality reduction
def reduce_dim_pca(train_x, val_x, variance_percent):
    pca = PCA()
    pca.fit(train_x)
    num_features = get_cumulative_contribution_index(pca.explained_variance_ratio_,
        percentage=variance_percent)

    pca = PCA(n_components=num_features)
    train_x = pd.DataFrame(pca.fit_transform(train_x))
    val_x   = pd.DataFrame(pca.transform(val_x))
    return train_x, val_x


def get_cumulative_contribution_index(ratios, percentage=0.8):
    '''
        Return index n such that the sum of the first n elements of the given ratios list have a sum
        greater than the percentage argument
    '''
    index = 0
    cumulative_percentage = 0
    while cumulative_percentage < percentage and index < len(ratios):
        cumulative_percentage += ratios[index]
        index += 1
    return index
