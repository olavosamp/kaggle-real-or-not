import pickle

import numpy as np

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
