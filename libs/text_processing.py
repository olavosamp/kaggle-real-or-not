import re
from pathlib import Path

import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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
    
    # Stem words
    if stemmer:
        def stem_word(x):
            return stemmer.stem(x)
        words = list(map(stem_word, words))

    return words


def process_dataset(train_set, test_set, result_dir=Path(commons.dataset_path),
    stemmer=PorterStemmer(), stopwords_set = set(stopwords.words("english"))):
    '''
        Apply text preprocessing and BoW coding to the dataset text features
    '''

    # Process text
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

    if result_dir:
        train_path = Path(result_dir) / "dataset_processed.csv"
        test_path  = Path(result_dir) / "test_processed.csv"    
        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)
    
    return train_set, test_set, vocabulary
