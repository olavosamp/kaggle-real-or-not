import time
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import scale

import libs.commons as commons
import libs.dataset as dataset

if __name__ == "__main__":
    train_path       = Path(commons.dataset_path) / "train.csv"
    test_path        = Path(commons.dataset_path) / "test.csv"
    random_seed      = 10
    epochs           = 500
    vocabulary_size  = 5000
    data_sample_size = 1.   # This should be 1 for training with the entire dataset
    identifier       = "sample_{:.0f}_vocabulary_size_{}".format(data_sample_size*100,
        vocabulary_size)

    # Create train and validation datasets
    train_x, val_x, train_y, val_y = dataset.create_dataset(train_path, test_path,
        random_seed=random_seed)

    print("Train set size: {}".format(len(train_y)))
    print("Validation set size: {}".format(len(val_y)))

    # Normalize the data to zero mean and unit variance
    train_x = scale(train_x)
    val_x   = scale(val_x)

    # Load model
    model = LogisticRegression(random_state=random_seed, max_iter=epochs)

    start = time.time()
    print("\nTraining model...")
    model.fit(train_x, train_y)
    elapsed = time.time() - start
    print("\nTraining complete\nElapsed time: {:.0f}s".format(elapsed))

    prediction = model.predict(val_x)

    result_f1  = f1_score(val_y, prediction)
    result_acc = accuracy_score(val_y, prediction)
    print("F1 Score: {:.2f}".format(result_f1))
    print("Accuracy: {:.2f}%".format(result_acc*100))
