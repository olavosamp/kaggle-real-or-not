from pathlib import Path

import nltk
import torch

import libs.models  as models
import libs.train   as train
import libs.commons as commons
from libs.dataset import TextDataset


if __name__ == "__main__":
    nltk.download('stopwords')

    train_processed_path = Path(commons.dataset_path) / "train_processed.csv"
    val_processed_path   = Path(commons.dataset_path) / "val_processed.csv"
    normalize            = True
    seed                 = 10
    vocabulary_size      = 5000 # TODO: Detect vocabulary size from data
    balance_data         = False
    balance_loss         = not(balance_data)
    freeze_conv          = False
    epochs               = 50
    batch_size           = 64
    learning_rate        = 0.001
    weight_decay         = 0.0001
    momentum             = 0.9
    step_size            = 20
    gamma                = 0.1
    data_sample_size     = 1.   # This should be 1 for training with the entire dataset
    identifier           = "sample_{:.0f}%_vocabulary_size_{}_loss-balance_{}_dataset-balance_{}_freeze_{}".format(
                            data_sample_size*100, vocabulary_size, balance_loss, balance_data, freeze_conv)

    # TODO: Finish train script
    # Define image transformations
    # image_transform = utils.resnet_transforms(defs.IMAGENET_MEAN, defs.IMAGENET_STD)

    print("\nInitializing dataloaders...")
    dataset = {}
    dataset["train"] = TextDataset(train_processed_path, target_column=commons.target_column_name,
        normalize=normalize, balance=balance_data)
    dataset["val"] = TextDataset(val_processed_path, target_column=commons.target_column_name,
        normalize=normalize, balance=False)
    num_features = dataset['train'].dataset.shape[1]

    print(f"Train features: {num_features}.")
    print(f"Train set size: {len(dataset['train'])}.")
    print(f"Validation set size: {len(dataset['val'])}.")

    # Load model
    # model = models.instantiate_resnet18_model(models.device, pretrained=True)
    model = models.FeedForwardNet(num_features, 140, 2)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #             momentum=momentum, weight_decay=weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                betas=(0.85, 0.99), weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                gamma=gamma)

    # results_folder = models.train_model(model, dataset, batch_size, optimizer, scheduler, epochs,
    #                     loss_balance=loss_balance, identifier=identifier,freeze_conv=freeze_conv)
    print("\nTraining model...")
    results_folder = train.train_feedforward_net(model, dataset, batch_size, optimizer, scheduler,
        epochs, loss_balance=balance_loss, identifier=identifier)
