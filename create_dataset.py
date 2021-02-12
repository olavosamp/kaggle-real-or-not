from pathlib import Path

import libs.commons as commons
from libs.dataset import create_dataset

seed                 = 10
train_path           = Path(commons.dataset_path) / "train.csv"
test_path            = Path(commons.dataset_path) / "test.csv"

# Create train and validation datasets
_, _, _, _ = create_dataset(train_path, test_path, random_seed=seed)
