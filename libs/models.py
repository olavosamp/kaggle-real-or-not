import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy  as np

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

if __name__ == "__main__":
    pass
