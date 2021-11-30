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

    activation = 'relu'
    epochs = 150
    batch_size = 64
    learning_rate = 3e-4
    loss = 'Cross-entropy'

    min_val = np.min(x_train)
    max_val = np.max(x_train)

    accuracies = {}

    for reg_coef in [0, 0.001, 0.01, 0.1, 1]:

        activation_fn = relu
        activation_fn_backwards = relu_backwards

        # Standardization to [0, 1] : greater than 0, better for the relu
        x_train = (x_train - min_val)/(max_val - min_val)
        x_test = (x_test - min_val)/(max_val - min_val)

        m = NeuralNetwork(activation_fn, activation_fn_backwards,
                          loss, units=128, verbose=True)
        metrics_history = m.train(x_train, y_train, x_test, y_test, epochs=epochs,
                                  batch_size=batch_size, learning_rate=learning_rate, reg_coef=reg_coef)
        train_loss_history, train_acc_history, test_loss_history, test_acc_history = metrics_history
        accuracies[reg_coef] = test_acc_history

    plt.figure()

    for reg_coef in [0, 0.001, 0.01, 0.1]:
        plt.plot(range(1, epochs+1), accuracies[reg_coef], label=reg_coef)

    plt.title("Accuracy over rounds")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
