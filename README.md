# batch-norm
An implementation of the technique of [batch normalization](https://arxiv.org/abs/1502.03167). The net is a 3-layer feedforward neural network whose 2 hidden layers have 60 neurons each. The default activations are tan-sigmoid for the hidden layers but log-sigmoid and ReLU activations are also available. The code for getting the MNIST dataset as Numpy arrays is taken from [here](https://github.com/datapythonista/mnist).

## Getting Started
Modify main.py with suitable values for 'total_epochs', the number of layers, and the number of neurons in each layer. Since a list with the number of neurons is passed, the number of layers can be varied if necessary too. On first-run of main.py the program will perform a one-time download of the MNIST dataset into /tmp/ clear the cache from this location if the download fails for any reason. A graph of the loss value and the test accuracy are created as ./loss.png and ./test.png respectively.

### Prerequisites
Python3\
NumPy\
Matplotlib - for graphing of the final data
