from pathlib import Path

import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

import libs.commons as commons

nltk.download("stopwords")

def process_text(text, stopwords_set=None, stemmer=None):
    '''
        Remove HTML tags, foreign characters, convert to lowercase and optionally remove stopwords.
        If Stopwords are to be removed, a set with the stopwords must be provided.
    '''
    text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9#]", " ", text) # Keep only alphanumeric characters and hashtags
    text = text.lower()
    words = set(text.split()) # Split string into words

    # Remove stopwords
    if stopwords_set:
        words = list(words - stopwords_set)
    # words = [w for w in words if w not in stopwords.words("english")]
    
    # Stem words with PorterStemmer
    if stemmer:
        def stem_word(x):
            return stemmer.stem(x)
        words = list(map(stem_word, words))

    return words


def process_dataset(train_set, test_set, cached_path=Path(commons.dataset_path) / 'train_processed.csv'):
    '''
        Apply text preprocessing and BoW coding to the dataset text features
    '''

    # if cached_path:
    #     if Path(cached_path).is_file():
    #         return pd.read_csv(cached_path)
    
    # Process text
    stopwords_set = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    train_proc_text = train_set['text'].apply(process_text, stopwords_set=stopwords_set,
        stemmer=stemmer)
    test_proc_text = test_set['text'].apply(process_text, stopwords_set=stopwords_set,
        stemmer=stemmer)

    # Get BoW matrix
    vectorizer = CountVectorizer(max_features=5000, preprocessor=lambda x: x, tokenizer=lambda x: x)
    features_train = vectorizer.fit_transform(train_proc_text).toarray()
    vocabulary = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names()

    # Apply the same vectorizer to the test set without training
    features_test = vectorizer.transform(test_proc_text).toarray()
    train_set = pd.concat([train_set, pd.DataFrame(data=features_train, columns=feature_names)],
        axis=1)
    test_set = pd.concat([test_set, pd.DataFrame(data=features_test, columns=feature_names)],
        axis=1)

    
    train_set.drop(columns=['id', 'keyword', 'location', 'text'], inplace=True)
    test_set.drop(columns=['id', 'keyword', 'location', 'text'], inplace=True)
    print("BoW matrix")
    print(train_set.head())
    print(train_set.shape)
    input()
    if cached_path:
        train_set.to_csv(cached_path)
    
    return train_set, test_set, vocabulary

train_path = Path(commons.dataset_path) / "train.csv"
test_path = Path(commons.dataset_path) / "test.csv"

train_set = pd.read_csv(train_path)
test_set = pd.read_csv(test_path)

# Preprocess and clean text features
train_set, test_set, vocabulary = process_dataset(train_set, test_set)

# print(train_proc.head())

# def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
#                          cache_dir=cache_dir, cache_file="bow_features.pkl"):
#     """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""
    
#     # If cache_file is not None, try to read from it first
#     cache_data = None
#     if cache_file is not None:
#         try:
#             with open(os.path.join(cache_dir, cache_file), "rb") as f:
#                 cache_data = joblib.load(f)
#             print("Read features from cache file:", cache_file)
#         except:
#             pass  # unable to read from cache, but that's okay
    
#     # If cache is missing, then do the heavy lifting
#     if cache_data is None:
#         # Fit a vectorizer to training documents and use it to transform them
#         # NOTE: Training documents have already been preprocessed and tokenized into words;
#         #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
#         vectorizer = CountVectorizer(max_features=vocabulary_size,
#                 preprocessor=lambda x: x, tokenizer=lambda x: x)  # already preprocessed
#         features_train = vectorizer.fit_transform(words_train).toarray()

#         # Apply the same vectorizer to transform the test documents (ignore unknown words)
#         features_test = vectorizer.transform(words_test).toarray()
        
#         # NOTE: Remember to convert the features using .toarray() for a compact representation
        
#         # Write to cache file for future runs (store vocabulary as well)
#         if cache_file is not None:
#             vocabulary = vectorizer.vocabulary_
#             cache_data = dict(features_train=features_train, features_test=features_test,
#                              vocabulary=vocabulary)
#             with open(os.path.join(cache_dir, cache_file), "wb") as f:
#                 joblib.dump(cache_data, f)
#             print("Wrote features to cache file:", cache_file)
#     else:
#         # Unpack data loaded from cache file
#         features_train, features_test, vocabulary = (cache_data['features_train'],
#                 cache_data['features_test'], cache_data['vocabulary'])
    
#     # Return both the extracted features as well as the vocabulary
#     return features_train, features_test, vocabulary
