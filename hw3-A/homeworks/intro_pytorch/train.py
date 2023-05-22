from typing import Dict, List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import problem


@problem.tag("hw3-A")
def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Performs training of a provided model and provided dataset.

    Args:
        train_loader (DataLoader): DataLoader for training set.
        model (nn.Module): Model to train.
        criterion (nn.Module): Callable instance of loss function, that can be used to calculate loss for each batch.
        optimizer (optim.Optimizer): Optimizer used for updating parameters of the model.
        val_loader (Optional[DataLoader], optional): DataLoader for validation set.
            If defined, if should be used to calculate loss on validation set, after each epoch.
            Defaults to None.
        epochs (int, optional): Number of epochs (passes through dataset/dataloader) to train for.
            Defaults to 100.

    Returns:
        Dict[str, List[float]]: Dictionary with history of training.
            It should have have two keys: "train" and "val",
            each pointing to a list of floats representing loss at each epoch for corresponding dataset.
            If val_loader is undefined, "val" can point at an empty list.

    Note:
        - Calculating training loss might expensive if you do it seperately from training a model.
            Using a running loss approach is advised.
            In this case you will just use the loss that you called .backward() on add sum them up across batches.
            Then you can divide by length of train_loader, and you will have an average loss for each batch.
        - You will be iterating over multiple models in main function.
            Make sure the optimizer is defined for proper model.
        - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
            You might find some examples/tutorials useful.
            Also make sure to check out torch.no_grad function. It might be useful!
        - Make sure to load the model parameters corresponding to model with the best validation loss (if val_loader is provided).
            You might want to look into state_dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    # initialize the optimizer
    optimizer_inst = optimizer(model.parameters(), lr=0.001)

    # Loop through all epochs and adjust weights for each epoch
    num_batches = len(train_loader)
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        train_loss_batch_average = 0
        val_loss_batch_average = 0
        for batch in range(num_batches):
            # access data from dataloader
            x_train, y_train = next(iter(train_loader))
            x_val, y_val = next(iter(val_loader))

            # zero the gradient function 
            optimizer_inst.zero_grad()

            # compute the loss for training and validation data
            train_loss_individual = criterion(model(x_train), y_train)
            
            # take gradient step for this epoch to update the model weight and bias
            train_loss_individual.backward()
            optimizer_inst.step()

            # add the train and validation loss values for each batch 
            train_loss_batch_average += train_loss_individual.detach().numpy()
            val_loss_batch_average += criterion(model(x_val), y_val).detach().numpy()

        # append the batch averaged loss values
        train_loss.append(train_loss_batch_average/num_batches)
        val_loss.append(val_loss_batch_average/num_batches)

    return {'train':train_loss, 'val':val_loss}


def plot_model_guesses(
    dataloader: DataLoader, model: nn.Module, title: Optional[str] = None
):
    """Helper function!
    Given data and model plots model predictions, and groups them into:
        - True positives
        - False positives
        - True negatives
        - False negatives

    Args:
        dataloader (DataLoader): Data to plot.
        model (nn.Module): Model to make predictions.
        title (Optional[str], optional): Optional title of the plot.
            Might be useful for distinguishing between MSE and CrossEntropy.
            Defaults to None.
    """
    with torch.no_grad():
        list_xs = []
        list_ys_pred = []
        list_ys_batch = []
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            list_xs.extend(x_batch.numpy())
            list_ys_batch.extend(y_batch.numpy())
            list_ys_pred.extend(torch.argmax(y_pred, dim=1).numpy())

        xs = np.array(list_xs)
        ys_pred = np.array(list_ys_pred)
        ys_batch = np.array(list_ys_batch)

        # True positive
        if len(ys_batch.shape) == 2 and ys_batch.shape[1] == 2:
            # MSE fix
            ys_batch = np.argmax(ys_batch, axis=1)
        idxs = np.logical_and(ys_batch, ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="o", c="green", label="True Positive"
        )
        # False positive
        idxs = np.logical_and(1 - ys_batch, ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="o", c="red", label="False Positive"
        )
        # True negative
        idxs = np.logical_and(1 - ys_batch, 1 - ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="x", c="green", label="True Negative"
        )
        # False negative
        idxs = np.logical_and(ys_batch, 1 - ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="x", c="red", label="False Negative"
        )

        if title:
            plt.title(title)
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.legend()
        plt.show()
