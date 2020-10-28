import numpy as np
import pandas as pd
from pathlib import Path

import libs.utils as utils
import libs.commons as commons
from libs.text_processing import process_dataset

random_seed = 10

train_path = Path(commons.dataset_path) / "train.csv"
test_path = Path(commons.dataset_path) / "test.csv"

train_set = pd.read_csv(train_path)
test_set = pd.read_csv(test_path)

# Preprocess and clean text features
train_set, test_set, vocabulary = process_dataset(train_set, test_set,
    result_dir=commons.dataset_path)

train_x, val_x, train_y, val_y = utils.split_train_val(train_set, train_size=0.8,
    random_seed=random_seed)

print("train_x: ", train_x.shape)
print("train_y: ", train_y.shape)
print("val_x: ", val_x.shape)
print("val_y: ", val_y.shape)
