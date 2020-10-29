import time
from pathlib import Path

import torch
import torchvision
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from lib.models      import model_pytorch
import libs.commons as commons
import libs.dataset as dataset

if __name__ == "__main__":
    train_path       = Path(commons.dataset_path) / "train.csv"
    test_path        = Path(commons.dataset_path) / "test.csv"
    normalize        = True
    random_seed      = 10
    vocabulary_size  = 5000
    dataset_balance  = True
    loss_balance     = not(dataset_balance)
    freeze_conv      = False
    batch_size       = 64
    learning_rate    = 0.001
    weight_decay     = 0.0001
    momentum         = 0.9
    epochs           = 5
    step_size        = 20
    gamma            = 0.1
    data_sample_size = 1.   # This should be 1 for training with the entire dataset
    identifier       = "sample_{:.0f}%_vocabulary_size_{}_loss-balance_{}_dataset-balance_{}_freeze_{}".format(
                            data_sample_size*100, vocabulary_size, loss_balance, dataset_balance, freeze_conv)

    # TODO: Finish train script
    # Define image transformations
    image_transform = utils.resnet_transforms(defs.IMAGENET_MEAN, defs.IMAGENET_STD)

    # Create train and validation datasets
    train_x, val_x, train_y, val_y = dataset.create_dataset(train_path, test_path,
        random_seed=random_seed)

    dataset = {}
    dataset["train"] = TextDataset(data_path, target_column=commons.target_column_name, normalize=normalize,
        balance=balance)
    dataset["val"] = TextDataset(data_path, target_column=commons.target_column_name, normalize=normalize,
        balance=False)

    print("Train set size: {}.".format(len(dataset["train"])))
    print("Validation set size: {}.".format(len(dataset["val"])))

    # Load model
    # resnet = torchvision.models.resnext50_32x4d(pretrained=True)
    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Linear(512, 2)
    model = resnet
    model.to(lib.model.device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #             momentum=momentum, weight_decay=weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                betas=(0.85, 0.99), weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                gamma=gamma)

    results_folder = model_pytorch.train_model(model, dataset, batch_size, optimizer, scheduler, epoch_number,
                        use_metadata, loss_balance=loss_balance, identifier=identifier,
                        freeze_conv=freeze_conv)

    vutils.plot_val_from_results(results_folder, dest_dir=None)
