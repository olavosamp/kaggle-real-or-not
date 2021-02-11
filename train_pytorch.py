import time
from pathlib import Path

import torch
import torchvision
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import libs.models   as models
import libs.commons as commons
from libs.dataset import TextDataset, create_dataset

if __name__ == "__main__":
    train_path           = Path(commons.dataset_path) / "train.csv"
    val_path             = Path(commons.dataset_path) / "val.csv"
    test_path            = Path(commons.dataset_path) / "test.csv"
    train_processed_path = Path(commons.dataset_path) / "train_processed.csv"
    val_processed_path   = Path(commons.dataset_path) / "val_processed.csv"
    normalize            = True
    random_seed          = 10
    vocabulary_size      = 5000
    balance              = True
    loss_balance         = not(balance)
    freeze_conv          = False
    batch_size           = 64
    learning_rate        = 0.001
    weight_decay         = 0.0001
    momentum             = 0.9
    epochs               = 5
    step_size            = 20
    gamma                = 0.1
    data_sample_size     = 1.   # This should be 1 for training with the entire dataset
    identifier           = "sample_{:.0f}%_vocabulary_size_{}_loss-balance_{}_dataset-balance_{}_freeze_{}".format(
                            data_sample_size*100, vocabulary_size, loss_balance, balance, freeze_conv)

    # TODO: Finish train script
    # Define image transformations
    # image_transform = utils.resnet_transforms(defs.IMAGENET_MEAN, defs.IMAGENET_STD)

    # Create train and validation datasets
    _, _, _, _ = create_dataset(train_path, test_path, random_seed=random_seed)

    dataset = {}
    dataset["train"] = TextDataset(train_processed_path, target_column=commons.target_column_name, normalize=normalize,
        balance=balance)
    dataset["val"] = TextDataset(val_processed_path, target_column=commons.target_column_name, normalize=normalize,
        balance=False)

    print("Train set size: {}.".format(len(dataset["train"])))
    print("Validation set size: {}.".format(len(dataset["val"])))

    # Load model
    # resnet = torchvision.models.resnext50_32x4d(pretrained=True)
    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Linear(512, 2)
    model = resnet
    model.to(models.device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #             momentum=momentum, weight_decay=weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                betas=(0.85, 0.99), weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                gamma=gamma)

    results_folder = models.train_model(model, dataset, batch_size, optimizer, scheduler, epochs,
                        loss_balance=loss_balance, identifier=identifier,freeze_conv=freeze_conv)
