if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    x_test, y_test = next(iter(dataloader))
    y_pred = model(x_test)
    return torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y_test, dim=1)).item()/y_test.shape[0]


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    n, d = dataset_train.tensors[0].shape
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True)

    # Set the fixed hyperparameters 
    num_epochs = 200
    
    # linear, one layer model
    model_linear = LinearLayer(dim_in=d, dim_out=2)

    # one linear hidden layer with sigmoid activation
    model_sigmoid = nn.Sequential(LinearLayer(dim_in=d, dim_out=2), SigmoidLayer())

    # one linear hidden layer with relu activation
    model_relu = nn.Sequential(LinearLayer(dim_in=d, dim_out=2), ReLULayer())

    # two linear hidden layers with sigmoid then relu activation
    model_sigmoid_then_relu = nn.Sequential(LinearLayer(dim_in=d, dim_out=2), SigmoidLayer(),
                                            LinearLayer(dim_in=2, dim_out=2), ReLULayer()) 
    
    # two linear hidden layers with relu then sigmoid activation
    model_relu_then_sigmoid = nn.Sequential(LinearLayer(dim_in=d, dim_out=2), ReLULayer(),
                                            LinearLayer(dim_in=2, dim_out=2), SigmoidLayer()) 
    
    # make a list of models
    models = [model_linear, model_sigmoid, model_relu, model_sigmoid_then_relu, model_relu_then_sigmoid]
    model_names = ['linear', 'sigmoid', 'relu', 'sigmoid then relu', 'relu then sigmoid']

    # loop through all models to compute the error 
    model_loss_mse = []
    for model in models:
        model_loss_mse.append(train(train_loader=train_loader, model=model, criterion=MSELossLayer(), 
                                optimizer=SGDOptimizer, val_loader=val_loader, epochs=num_epochs))

     # plot the train and validation loss 
    model_colors = ['k', 'r', 'b', 'g', 'y']
    fig, ax = plt.subplots()
    for n in range(len(models)):
        ax.plot(model_loss_mse[n]['train'], label=f'Train - {model_names[n]}', color=model_colors[n], linestyle='dashed')
        ax.plot(model_loss_mse[n]['val'], label=f'Validation - {model_names[n]}', color=model_colors[n])
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    plt.show()

    # save the model loss and model
    model_history = {}
    for n in range(len(models)):
        model_history[model_names[n]] = model_loss_mse[n]
        model_history[model_names[n]]['model'] = models[n]

    return model_history
    
@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), 
                                  torch.from_numpy(to_one_hot(y)).float())
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val)).float()
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test)).float()
    )

    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    
    # Plot model guesses using the model 
    test_loader = DataLoader(dataset_test, batch_size=x_test.shape[0], shuffle=True)
    plot_model_guesses(test_loader, mse_configs['relu then sigmoid']['model'], 
                       'Two Hidden Layers - ReLu then Sigmoid Activation NN')
    
    # compute accuracy of the best model
    acc = accuracy_score(mse_configs['relu then sigmoid']['model'], test_loader)
    print(f'The accuracy of the relu then sigmoid model is {acc}.')

def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()

