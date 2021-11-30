
from data import lecture_cifar, decoupage_donnees, one_hot_encoding, make_batches, descripteur_hog, cross_validation
from activations import sigmoid, sigmoid_backwards, relu, relu_backwards, linear, linear_backwards
from neuralnetwork import NeuralNetwork, NeuralNetworkTensorFlowLike, NeuralNetworkTwoHiddenLayers
import numpy as np
import matplotlib.pyplot as plt

path = "./cifar-10-python/cifar-10-batches-py"

x, y = lecture_cifar(path, 5)
NUM_CLASSES = max(y) + 1
x = x/np.max(x)
y = one_hot_encoding(y, NUM_CLASSES)
x_train, y_train, x_test, y_test = decoupage_donnees(x, y, 0.8)


if __name__ == '__main__':
    PATH = "./cifar-10-python/cifar-10-batches-py"
    NUM_CLASSES = 10

    use_hog = True
    activation = 'relu'
    loss = 'Cross-entropy'
    epochs = 100
    batch_size = 128
    learning_rate = 3e-3
    reg_coef = 0.001

    min_val = np.min(x_train)
    max_val = np.max(x_train)

    if activation == 'sigmoid':
        activation_fn = sigmoid
        activation_fn_backwards = sigmoid_backwards

        # Standardization to [-1, -1] : the center of the sigmoid
        x_train = -1 + 2*(x_train - min_val)/(max_val - min_val)
        x_test = -1 + 2*(x_test - min_val)/(max_val - min_val)

    elif activation == 'relu':
        activation_fn = relu
        activation_fn_backwards = relu_backwards

        # Standardization to [0, 1] : greater than 0, better for the relu
        x_train = (x_train - min_val)/(max_val - min_val)
        x_test = (x_test - min_val)/(max_val - min_val)

    else:
        raise ValueError("Undefined activation")

    # Load data and preprocess it
    x, y = lecture_cifar(PATH, 5)
    y = one_hot_encoding(y, NUM_CLASSES)
    if use_hog:
        print("Processing HOG features")
        x = descripteur_hog(x)
        print("Done")
    x_train, y_train, x_test, y_test = decoupage_donnees(x, y, 0.8)

    # Training
    results = {}

    units = []

    for learning_rate in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        def score_function(x, y, x_val, y_val):
            m = NeuralNetwork(
                activation_fn, activation_fn_backwards, loss, units=256, verbose=False)
            metrics_history = m.train(x, y, x_val, y_val, epochs=epochs,
                                      batch_size=batch_size, learning_rate=learning_rate, reg_coef=reg_coef)
            (_, train_acc_history, _, test_acc_history) = metrics_history
            max_train = max(train_acc_history)
            max_test = max(test_acc_history)

            return max_train, max_test

        train, test = cross_validation(score_function, x, y, 5)
        results[learning_rate] = test

    print(results)
