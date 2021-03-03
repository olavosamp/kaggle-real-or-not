import re
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

import libs.commons as commons
import libs.utils as utils
from sklearn.model_selection import train_test_split

class TextDataset(Dataset):
    def __init__(self, dataset_path, target_column, normalize, balance):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.balance = balance
        self.normalize = normalize

        self.read_dataset()
        self.length = len(self.target)

        # If the dataset will be balanced, change length to the double of the largest class
        positive_len = self.target.sum() # Count positive elements (ones)
        if self.balance and positive_len*2 != self.length:
            if positive_len > self.length:
                self.length = 2 * positive_len
            else:
                self.length = 2 * (self.length - positive_len)

        # # Normalize dataset.
        # if self.normalize:
        #     self.dataset.loc[:,:] = (scale_dataset(self.dataset.loc[:,:]))

    def read_dataset(self):
        '''Reads csv file to feature DataFrame and target ndarray'''
        assert Path(self.dataset_path).is_file(), "Dataset path does not exists."

        dataset = pd.read_csv(self.dataset_path)
        self.target = dataset.loc[:, self.target_column].values
        self.dataset = dataset.drop(columns=[self.target_column])

    def imbalance_ratio(self):
        '''Compute the ratio between the negative and positive classes.
        If the dataset is artificially balanced, this ratio is 1.
        '''
        if self.balance:
            return 1.0
        else:
            return (1 - self.target).sum() / self.target.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ''' Balance the dataset by using the positive samples multiple times.
        Indices between the original size and 2 times the number of negative
        samples are reassigned to positive sample indices.
        '''
        # if self.balance:
        #     if idx < 0:
        #         idx = self.length + idx
        #     if idx >= len(self.target):
        #         idx = (idx - len(self.target)) % self.target.sum()
        #         idx = (self.target != 0).nonzero()[0][idx].item()
            # target = torch.tensor(self.target-[idx])

        # Convert data to torch tensors
        entry  = torch.tensor(self.dataset.loc[idx, :].values)
        target = torch.tensor(self.target[idx])

        return entry, target


def scale_dataset(train_x, val_x, test_x):
    '''Compute mean and std from the train set and use them to scale val and test sets as well'''
    scaler = StandardScaler()
    train_x.loc[:,:] = scaler.fit_transform(train_x.loc[:,:])
    val_x.loc[:,:]   = scaler.transform(val_x.loc[:,:])
    test_x.loc[:,:]  = scaler.transform(test_x.loc[:,:])
    return train_x, val_x, test_x


def remove_empty_features(train_set, test_set, verbose=False):
    '''Drop feature columns that have zero variance'''
    drop_train = set(train_set.columns[train_set.std(axis=0) == 0])
    drop_test  = set(test_set.columns[test_set.std(axis=0) == 0])
    drop_columns = drop_train.intersection(drop_test)
    if verbose:
        print("\nRemoving non-informative features...\n", list(drop_columns))

    train_set = train_set.drop(columns=drop_columns)
    test_set  = test_set.drop(columns=drop_columns)
    return train_set, test_set


def create_dataset(train_path, test_path, seed=10, standardize=True, pca_ratio=0.90,
     save_dir=commons.dataset_path):
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)

    # Preprocess and clean text features
    train_set, test_set, vocabulary = process_dataset(train_set, test_set,
        result_dir=commons.dataset_path, stopwords=stopwords.words("english"))

    train_set, test_set = remove_empty_features(train_set, test_set, verbose=True)
    vocabulary = set(vocabulary).intersection(train_set.columns)

    # Split data in train and val sets
    print("\nSplitting dataset...")
    train_x, val_x, train_y, val_y = split_train_val(train_set, train_size=0.8,
        seed=seed)

    if standardize:
        print("\nStandardizing dataset...")
        train_x, val_x, test_set = scale_dataset(train_x, val_x, test_set)

    if pca_ratio < 1.:
        print(f"\nPerforming dimensionality reduction...\nKeep {pca_ratio*100:.2f}% variance")
        train_x, val_x = utils.reduce_dim_pca(train_x, val_x, pca_ratio)

    # Save data to csv
    if save_dir:
        print("\nSaving dataset to file...")
        train_set = train_x.copy()
        val_set   = val_x.copy()
        train_set[commons.target_column_name] = train_y
        val_set[commons.target_column_name]   = val_y

        train_set.to_csv(Path(save_dir) / "train_processed.csv", index=False)
        val_set.to_csv(Path(save_dir)   / "val_processed.csv", index=False)
        test_set.to_csv(Path(save_dir)  / "test_processed.csv", index=False)

    # Return data anyway for sklearn models
    return train_x, val_x, train_y, val_y


def split_train_val(train_set, train_size=0.8, seed=None):
    train_y = train_set.loc[:, commons.target_column_name]
    train_x = train_set.drop(columns=commons.target_column_name)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=train_size,
        random_state=seed)

    return train_x, val_x, train_y, val_y


def process_text(text, stopwords=None, stemmer=None):
    '''
        Remove HTML tags, foreign characters, convert to lowercase and optionally remove stopwords.
        If Stopwords are to be removed, a set with the stopwords must be provided.
    '''
    text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9#]", " ", text) # Keep only alphanumeric characters and hashtags
    text = text.lower()
    words = text.split() # Split string into words

    # Remove stopwords
    if stopwords:
        words = list(set(words) - set(stopwords))
        # words = [w for w in words if w not in stopwords.words("english")]

    # Stem words
    if stemmer:
        def stem_word(x):
            return stemmer.stem(x)
        words = list(map(stem_word, words))

    return words


def bow_matrix(train_text, test_text, max_features, load_path=None, save_path=None):
    vectorizer = CountVectorizer(max_features=5000, preprocessor=lambda x: x, tokenizer=lambda x: x)

    if load_path:
        vectorizer.vocabulary_ = utils.load_pickle(load_path)
        features_train = vectorizer.transform(train_text).toarray()
    else:
        features_train = vectorizer.fit_transform(train_text).toarray()

    vocabulary    = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names()

    features_test = vectorizer.transform(test_text).toarray()

    new_train_df = pd.DataFrame(data=features_train, columns=feature_names)
    new_test_df  = pd.DataFrame(data=features_test, columns=feature_names)

    if save_path:
        utils.save_pickle(vocabulary, save_path)

    return new_train_df, new_test_df, vocabulary


def process_dataset(train_set, test_set, result_dir=Path(commons.dataset_path),
    stemmer=PorterStemmer(), stopwords=None, verbose=True):
    '''Apply text preprocessing and BoW coding to the dataset text features'''

    if verbose:
        print("\nCleaning text...")
    train_proc_text = train_set['text'].apply(process_text, stopwords=stopwords,
        stemmer=stemmer)
    test_proc_text = test_set['text'].apply(process_text, stopwords=stopwords,
        stemmer=stemmer)

    if verbose:
        print("\nAssembling Bag of Words matrix...")
    features_path = Path(commons.experiments_path) / "vectorizer_params.pickle"
    new_train_df, new_test_df, vocabulary = bow_matrix(
        train_proc_text, test_proc_text, 5000, load_path=features_path, save_path=features_path)

    train_set[commons.target_column_name] = train_set['target']
    train_set.drop(columns=['id', 'keyword', 'location', 'text', 'target'], inplace=True)
    test_set.drop(columns=['id', 'keyword', 'location', 'text'], inplace=True)

    train_set = pd.concat([train_set, new_train_df], axis=1)
    test_set = pd.concat([test_set, new_test_df], axis=1)

    if result_dir:
        train_path = Path(result_dir) / "dataset_processed.csv"
        test_path  = Path(result_dir) / "test_processed.csv"
        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

    return train_set, test_set, vocabulary
