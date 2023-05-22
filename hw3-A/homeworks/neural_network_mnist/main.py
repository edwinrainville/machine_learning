# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.weight_0 = Parameter(Uniform(-1/math.sqrt(d), 1/math.sqrt(d)).sample((h, d)))
        self.bias_0 = Parameter(Uniform(-1/math.sqrt(d), 1/math.sqrt(d)).sample((h,)))
        self.weight_1 = Parameter(Uniform(-1/math.sqrt(h), 1/math.sqrt(h)).sample((k, h)))
        self.bias_1 = Parameter(Uniform(-1/math.sqrt(h), 1/math.sqrt(h)).sample((k,)))
        
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.
        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden_layer = torch.matmul(x, torch.transpose(self.weight_0, 0, 1)) + self.bias_0
        hidden_layer_relu = relu(hidden_layer)
        return torch.matmul(hidden_layer_relu, torch.transpose(self.weight_1, 0, 1)) + self.bias_1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        # layer 0
        self.weight_0 = Parameter(Uniform(-1/math.sqrt(d), 1/math.sqrt(d)).sample((h0, d)))
        self.bias_0 = Parameter(Uniform(-1/math.sqrt(d), 1/math.sqrt(d)).sample((h0,)))

        # layer 1
        self.weight_1 = Parameter(Uniform(-1/math.sqrt(h0), 1/math.sqrt(h0)).sample((h1, h0)))
        self.bias_1 = Parameter(Uniform(-1/math.sqrt(h0), 1/math.sqrt(h0)).sample((h1,)))

        # layer 2
        self.weight_2 = Parameter(Uniform(-1/math.sqrt(h1), 1/math.sqrt(h1)).sample((k, h1)))
        self.bias_2 = Parameter(Uniform(-1/math.sqrt(h1), 1/math.sqrt(h1)).sample((k,)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden_layer_0 = torch.matmul(x, torch.transpose(self.weight_0, 0, 1)) + self.bias_0
        hidden_layer_0_relu = relu(hidden_layer_0)
        hidden_layer_1 = torch.matmul(hidden_layer_0_relu, torch.transpose(self.weight_1, 0, 1)) + self.bias_1
        hidden_layer_1_relu = relu(hidden_layer_1)
        return torch.matmul(hidden_layer_1_relu, torch.transpose(self.weight_2, 0, 1)) + self.bias_2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    # instance of optimizer
    optimizer = Adam(model.parameters(), lr=5e-3)

    loss = []
    accuracy = 0
    # train model until 99% accurate
    while accuracy < 0.99:
        # get x and y from the train_loader
        x, y = next(iter(train_loader))
        
        # evaluate loss and take next step
        optimizer.zero_grad()
        y_pred = model(x)
        loss_in_epoch = cross_entropy(y_pred, y)
        loss_in_epoch.backward()
        optimizer.step()

        # append loss, compute accuracy, and index up epoch
        loss.append(loss_in_epoch.item())
        accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y).item()/len(y)

    return loss

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # build train dataloader
    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    # create model instances
    f1_model = F1(h=64, d=784, k=10)

    # train the model
    f1_model_loss = train(f1_model, Adam, train_loader)

    # plot the loss as a function of epochs
    fig, ax = plt.subplots()
    ax.plot(f1_model_loss)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('F1 training Loss')
    plt.show()

    # run model on the test data and evaluate loss and accuracy
    y_test_pred_f1 = f1_model(x_test)
    f1_test_loss = cross_entropy(y_test_pred_f1, y_test).item()
    f1_test_accuracy = torch.sum(torch.argmax(y_test_pred_f1, dim=1) == y_test).item()/len(y_test)
    print(f'F1 model test loss is {f1_test_loss} and test accuracy is {f1_test_accuracy}')

    # create model instances
    f2_model = F2(h0=32, h1=32, d=784, k=10)

    # train the model
    f2_model_loss = train(f2_model, Adam, train_loader)

    # plot the loss as a function of epochs
    fig, ax = plt.subplots()
    ax.plot(f2_model_loss)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('F2 training Loss')
    plt.show()

    # run model on the test data and evaluate loss and accuracy
    y_test_pred_f2 = f2_model(x_test)
    f2_test_loss = cross_entropy(y_test_pred_f2, y_test).item()
    f2_test_accuracy = torch.sum(torch.argmax(y_test_pred_f2, dim=1) == y_test).item()/len(y_test)
    print(f'F2 model test loss is {f2_test_loss} and test accuracy is {f2_test_accuracy}')

    # Compute total parameters for each model
    k = 10
    d = 784
    h = 64
    f1_total_params = ((h * d) + h) + ((k*h) + k)
    print(f'Total number of parameters in f1 model is {f1_total_params}')
    h0 = 32
    h1 = 32
    f2_total_params = ((h0*d) + h0) + ((h1*h0) + h1) + ((k*h1) + k)
    print(f'Total number of parameters in f2 model is {f2_total_params}')

if __name__ == "__main__":
    main()
