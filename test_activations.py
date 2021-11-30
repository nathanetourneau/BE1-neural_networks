from data import lecture_cifar, decoupage_donnees, one_hot_encoding, make_batches, descripteur_hog, cross_validation
from activations import sigmoid, sigmoid_backwards, relu, relu_backwards, linear, linear_backwards, tanh, tanh_backwards
from neuralnetwork import NeuralNetwork, NeuralNetworkTensorFlowLike, NeuralNetworkTwoHiddenLayers
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    PATH = "./cifar-10-python/cifar-10-batches-py"
    NUM_CLASSES = 10

    # Load data and preprocess it
    x, y = lecture_cifar(PATH, 5)
    y = one_hot_encoding(y, NUM_CLASSES)

    use_hog = True

    if use_hog:
        print("Processing HOG features")
        x = descripteur_hog(x)
        print("Done")
    x_train, y_train, x_test, y_test = decoupage_donnees(x, y, 0.8)

    loss = 'MSE'
    epochs = 100
    batch_size = 16
    learning_rate = 1e-3
    reg_coef = 0

    min_val = np.min(x_train)
    max_val = np.max(x_train)

    accuracies = {}

    for activation in ['sigmoid', 'relu', 'tanh', 'linear']:
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

        elif activation == 'tanh':
            activation_fn = tanh
            activation_fn_backwards = tanh_backwards

        elif activation == 'linear':
            activation_fn = linear
            activation_fn_backwards = linear_backwards

        else:
            raise ValueError("Undefined activation")

        m = NeuralNetwork(activation_fn, activation_fn_backwards,
                          loss, units=128, verbose=True)
        metrics_history = m.train(x_train, y_train, x_test, y_test, epochs=epochs,
                                  batch_size=batch_size, learning_rate=learning_rate, reg_coef=reg_coef)
        train_loss_history, train_acc_history, test_loss_history, test_acc_history = metrics_history
        accuracies[activation] = test_acc_history

    plt.figure()
    for activation in ['sigmoid', 'relu', 'tanh', 'linear']:
        plt.plot(range(1, epochs+1), accuracies[activation], label=activation)
    plt.title("Accuracy over rounds")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
