from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import libs.commons as commons

def split_train_val(train_set, train_size=0.8, random_seed=None, result_dir=commons.dataset_path):
    train_y = train_set.loc[:, 'target']
    train_x = train_set.drop(columns='target')

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.8,
        random_state=random_seed)

    train_set = train_x.copy()
    val_set   = val_x.copy()

    train_set['target'] = train_y
    val_set['target']   = val_y

    if result_dir:
        train_path = Path(result_dir) / "train_processed.csv"
        val_path  = Path(result_dir) / "val_processed.csv"    
        train_set.to_csv(train_path, index=False)
        val_set.to_csv(val_path, index=False)

    return train_x, val_x, train_y, val_y
