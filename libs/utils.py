import pickle

## Pickle functions
def save_pickle(object, filePath):
    with open(filePath, 'wb') as handle:
        pickle.dump(object, handle)


def load_pickle(filePath):
    with open(filePath, 'rb') as handle:
        pickled_data = pickle.load(handle)
    return pickled_data
