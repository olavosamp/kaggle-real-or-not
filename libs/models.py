import os
import time
import uuid

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy  as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm

import libs.dataset
import libs.commons as commons

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForwardNet(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.

    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    def __init__(self, input_features, hidden_dim, output_dim, dropout_prob=0.):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.drop = nn.Dropout(self.dropout_prob)  # dropout with 30% prob

        # define all layers, here
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        x = x.view(-1, self.input_features)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.drop(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class EarlyStop:
    '''Early stopping monitor for loss minimization'''
    def __init__(self, patience=8, tol=1e-4):
        self.counter = 0
        self.patience = patience
        self.tol = tol
        self.best_loss = 999

    def step(self, loss):
        if self.best_loss - loss >  self.tol:
            self.counter = 0
            self.best_loss = loss
        else:
            self.counter += 1
        
        print(self.counter)
        
        return self.check_early_stop()

    def check_early_stop(self):
        if self.counter >= self.patience:
            return True # Stop
        else:
            return False # Do not stop


class MetricTracker:
    def __init__(self, metrics=[], threshold=0.5):
        self.metrics = metrics
        self.columns = ["epoch", "phase", "loss"] + metrics
        self.results_df = pd.DataFrame(self.columns)
        self.threshold = threshold
        self.time_start = None

    @staticmethod
    def calculate_accuracy(target, prediction, num_samples):
        correct = target == prediction
        return np.sum(correct) / num_samples

    @staticmethod
    def calculate_f1_score(target, prediction):
        return sklearn.metrics.f1_score(target, prediction)

    @staticmethod
    def calculate_roc_auc(target, confidence):
        return sklearn.metrics.roc_auc_score(target, confidence)

    def add(self, epoch, phase, target, confidence, loss, num_samples):
        prediction = confidence > self.threshold
        loss = loss / num_samples

        epoch_results = {"epoch": epoch, "phase": phase, "loss": loss}
        if "accuracy" in self.metrics:
            epoch_results["accuracy"] = self.calculate_accuracy(target, prediction, num_samples)
        if "f1_score" in self.metrics:
            epoch_results["f1_score"] = self.calculate_f1_score(target, prediction)
        if "roc_auc" in self.metrics:
            epoch_results["roc_auc"] = self.calculate_roc_auc(target, confidence)
        if "seconds" in self.metrics:
            epoch_results["seconds"] = self.time()
        self.results_df.append(epoch_results, sort=False, ignore_index=True)

    def time(self):
        if self.time_start is None:
            self.time_start = time.time()
            return None
        else:
            elapsed = time.time() - self.time_start
            self.time_start = None
            return elapsed

    @callable
    def last_result(self):
        return self.results_df.loc[self.results_df.index[-1], :]


def instantiate_resnet18_model(device, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    resnet.fc = torch.nn.Linear(512, 2)
    resnet.to(device)
    return resnet


def freeze_convolutional_resnet(model, freeze):
    '''
        Freezes or unfreezes the model convolutional layers. Assumes the passed model
        has a method named fc that corresponds to the only layer the user wishes to
        keep unfrozen.
        Argument:
            freeze: bool
                Pass True to freeze and False to unfreeze layers.
    '''
    # Freeze (or not) convolutional layers
    for child in model.children():
        for param in child.parameters():
            # Freeze == True sets requires_grad = False: freezes layers
            param.requires_grad = not(freeze)

    # Don't freeze FC layer
    for param in model.fc.parameters():
        param.requires_grad = True


def load_model(model, weight_path, device=device, eval=True):
    '''
        Load model weights in corresponding model object.

        model: instantiated model object
            Model into which load the weights.
        
        weight_path: string filepath
            Filepath of the weights.
    '''
    weight_path = str(weight_path)
    if not os.path.isfile(weight_path):
        raise ValueError("Invalid weight filepath.")

    checkpoint = torch.load(weight_path)#, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def predict(model, dataloader, device=device, threshold=None, use_metadata=True):
    '''
        Performs inference on the input data using a Pytorch model.
        model: model object
            Model which will perform inference on the data. Must support
            inference through syntax
                 result = model(input)
        data: collection of model inputs
            A collection, such as a list, of model inputs. Each element of data
            must match the required arguments of model.

        threshold: None or float (optional)
            Decision threshold to evaluate model outputs.
            
            If None, result will be an array of floats representing model confidence
            for each input.
             
            If float, result will be an array of zeros and ones. Each input will be
            converted to one if model confidence is greater than threshold and zero
            if lesser.

        Returns:
        result: array of float
    '''
    model.to(device)
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    result_list = []
    for image, metadata in tqdm(dataloader):
        image    = image.to(device)
        metadata = metadata.to(device)

        output = model(image)

        confidence = softmax(output).detach().cpu().numpy()[:, 1]
        result_list.append(confidence)

    return result_list


def train_feedforward_net(model, dataset, batch_size, optimizer, scheduler, num_epochs,
        loss_balance=True, identifier=None, device=device):
    print("\nUsing device: ", device)
    device_params = {"device": device, "dtype": torch.float64}
    model.to(**device_params)

    tracked_metrics = ["accuracy", "f1", "auc", "seconds"]
    early_stop = EarlyStop(tol=1e-5)
    metrics = MetricTracker(metrics=tracked_metrics)

    # Create unique identifier for this experiment.
    if identifier is None:
        identifier = str(uuid.uuid4())
    else:
        identifier = str(identifier) + "_" + str(uuid.uuid4())
    phase_list = ("train", "val")

    # Setup experiment paths
    experiment_dir = Path(commons.experiments_path) / str(identifier)
    weights_folder  = experiment_dir / "weights" 
    commons.create_folder(weights_folder)

    # Instantiate loss and softmax.
    if loss_balance:
        weight = [1.0, dataset["train"].imbalance_ratio()]
        weight = torch.tensor(weight).to(**device_params)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    # Define data loaders.
    data_loader = {phase: torch.utils.data.DataLoader(
        dataset[phase], batch_size=batch_size, shuffle=True, num_workers=4) for phase in phase_list}

    i = 0
    while i <= num_epochs and not early_stop.check_early_stop():
        print("\nEpoch: {}/{}".format(i+1, num_epochs))
        phase_loss = 0
        for phase in phase_list:
            print("\n{} phase: ".format(str(phase).capitalize()))

            # Set model to training or evalution mode according to the phase.
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_target = []
            epoch_confidence = []
            metrics.time()
            # Iterate over the dataset.
            for entry, target  in tqdm(data_loader[phase]):
                # Update epoch target list to compute AUC(ROC) later.
                epoch_target.append(target.numpy())

                # Load samples to device.
                entry = entry.to(**device_params)
                target = target.to(device, dtype=torch.int64)

                # Set gradients to zero.
                optimizer.zero_grad()

                # Calculate gradients only in the training phase.
                with torch.set_grad_enabled(phase=="train"):
                    output = model(entry)
                    loss = cross_entropy_loss(output, target)
                    confidence = softmax(output).detach().cpu().numpy()[:, 1]

                    # Backward gradients and update weights if training.
                    if phase=="train":
                        loss.backward()
                        optimizer.step()

                # Update epoch loss and epoch confidence list.
                phase_loss += loss.item()
                epoch_confidence.append(confidence)

            if phase == "train":
                scheduler.step()

            # Compute epoch loss, accuracy and AUC(ROC).
            num_samples = len(dataset[phase])
            epoch_target = np.concatenate(epoch_target, axis=0)
            epoch_confidence = np.concatenate(epoch_confidence, axis=0) # List of batch confidences
            metrics.add(i+1, phase, epoch_target, epoch_confidence, phase_loss, num_samples)

            time_string = time.strftime("%H:%M:%S", time.gmtime(metrics.last_result["seconds"]))
            print("Epoch complete in ", time_string)
            print("{} loss: {:.4f}".format(phase, metrics.last_result["loss"]))
            print("{} accuracy: {:.2f}%".format(phase, metrics.last_result["accuracy"]))
            print("{} F1: {:.4f}".format(phase, metrics.last_result["f1"]))
            print("{} area under ROC curve: {:.4f}".format(phase, metrics.last_result["auc"]))

        # Save metrics
        results_path = experiment_dir / "epoch_{}_results.json".format(i+1)
        metrics.results_df.to_csv(results_path, index=False)

        i += 1
        early_stop.step(metrics.last_result["loss"])
    return results_path.parent


def train_model(model, dataset, batch_size, optimizer, scheduler, num_epochs, loss_balance=True,
                identifier=None, freeze_conv=False):
    # Create unique identifier for this experiment.
    if identifier is None:
        identifier = str(uuid.uuid4())
    else:
        identifier = str(identifier) + "_" + str(uuid.uuid4())
    phase_list = ("train", "val")

    # Setup experiment paths
    experiment_dir = Path(commons.experiments_path) / str(identifier)
    weights_folder  = experiment_dir / "weights" 
    commons.create_folder(weights_folder)

    freeze_convolutional_resnet(model, freeze_conv)

    print("Using device: ", device)

    # Instantiate loss and softmax.
    if loss_balance:
        weight = [1.0, dataset["train"].imbalance_ratio()]
        weight = torch.tensor(weight).to(device)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    # Define data loaders.
    data_loader = {x: torch.utils.data.DataLoader(dataset[x],
        batch_size=batch_size, shuffle=True, num_workers=4)
        for x in phase_list}

    # Measures that will be computed later.
    tracked_metrics = ["epoch", "phase", "loss", "accuracy", "auc", "seconds"]
    epoch_auc       = {x: np.zeros(num_epochs) for x in phase_list}
    epoch_loss      = {x: np.zeros(num_epochs) for x in phase_list}
    epoch_accuracy  = {x: np.zeros(num_epochs) for x in phase_list}
    results_df      = pd.DataFrame()

    for i in range(num_epochs):
        print("\nEpoch: {}/{}".format(i+1, num_epochs))
        results_dict = {metric: [] for metric in tracked_metrics}
        for phase in phase_list:
            print("\n{} phase: ".format(str(phase).capitalize()))

            # Set model to training or evalution mode according to the phase.
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_target = []
            epoch_confidence = []
            epoch_seconds = time.time()
            # Iterate over the dataset.
            for entry, target  in tqdm(data_loader[phase]):
                # Update epoch target list to compute AUC(ROC) later.
                epoch_target.append(target.numpy())

                # Load samples to device.
                entry = entry.to(device)
                target = target.to(device)

                # Set gradients to zero.
                optimizer.zero_grad()

                # Calculate gradients only in the training phase.
                with torch.set_grad_enabled(phase=="train"):
                    output = model(entry)
                    loss = cross_entropy_loss(output, target)
                    confidence = softmax(output).detach().cpu().numpy()[:, 1]

                    # Backward gradients and update weights if training.
                    if phase=="train":
                        loss.backward()
                        optimizer.step()

                # Update epoch loss and epoch confidence list.
                epoch_loss[phase][i] += loss.item() * image.size(0)
                epoch_confidence.append(confidence)

            if phase == "train":
                scheduler.step()

            # Compute epoch loss, accuracy and AUC(ROC).
            sample_number = len(dataset[phase])
            epoch_target = np.concatenate(epoch_target, axis=0)
            epoch_confidence = np.concatenate(epoch_confidence, axis=0) # List of batch confidences
            epoch_loss[phase][i] /= sample_number
            epoch_correct = epoch_target == (epoch_confidence > 0.5)
            epoch_accuracy[phase][i] = (epoch_correct.sum() / sample_number)
            epoch_auc[phase][i] = sklearn.metrics.roc_auc_score(epoch_target,
                                                                epoch_confidence)
            epoch_seconds = time.time() - epoch_seconds

            time_string   = time.strftime("%H:%M:%S", time.gmtime(epoch_seconds))
            print("Epoch complete in ", time_string)
            print("{} loss: {:.4f}".format(phase, epoch_loss[phase][i]))
            print("{} accuracy: {:.4f}".format(phase, epoch_accuracy[phase][i]))
            print("{} area under ROC curve: {:.4f}".format(phase, epoch_auc[phase][i]))

            # Collect metrics in a dictionary
            results_dict["epoch"].append(i+1) # Epochs start at 1
            results_dict["phase"].append(phase)
            results_dict["loss"].append(epoch_loss[phase][i])
            results_dict["accuracy"].append(epoch_accuracy[phase][i])
            results_dict["auc"].append(epoch_auc[phase][i])
            results_dict["seconds"].append(epoch_seconds)

        # Save metrics to DataFrame
        results_df = results_df.append(pd.DataFrame(results_dict), sort=False, ignore_index=True)

        # Save model
        weights_path = weights_folder / "resnet18_epoch_{}_{}.pth".format(i+1, identifier)
        results_path = experiment_dir / "epoch_{}_results.json".format(i+1)
        torch.save(model.state_dict(), weights_path)
        results_df.to_csv(results_path, index=False)

    return results_path.parent

if __name__ == "__main__":
    pass
