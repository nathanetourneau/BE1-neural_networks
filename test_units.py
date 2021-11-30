
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
    epochs = 70
    batch_size = 256
    learning_rate = 3e-3
    reg_coef = 0.01

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

    """
    
    
    """
    m = NeuralNetworkTwoHiddenLayers(activation_fn, activation_fn_backwards,
                                     loss, units1=128, units2=64, verbose=True)
    metrics_history = m.train(x_train, y_train, x_test, y_test, epochs=epochs,
                              batch_size=batch_size, learning_rate=learning_rate, reg_coef=reg_coef)

    use_hog = True
    activation = 'relu'
    loss = 'Cross-entropy'
    epochs = 100
    batch_size = 64
    learning_rate = 3e-3
    reg_coef = 0.01
    results = []

    units_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for units in units_list:
        print(f'Units : {units}')
        m = NeuralNetwork(
            activation_fn, activation_fn_backwards, loss, units=units, verbose=True)
        metrics_history = m.train(x_train, y_train, x_test, y_test, epochs=epochs,
                                  batch_size=batch_size, learning_rate=learning_rate, reg_coef=reg_coef)
        (_, _, _, test_acc_history) = metrics_history
        max_test = max(test_acc_history)
        results.append(max_test)

    plt.figure()
    plt.plot(units_list, results)
    plt.title("Test accuracy after 100 epoch, for various units number")
    plt.xlabel("Number of units (log scale)")
    plt.ylabel("Test accuracy after 100 epoch")
    plt.xscale('log', base=2)
    plt.show()

    if False:
        plt.figure()
        plt.plot(range(1, epochs+1), train_loss_history, label='Train loss')
        plt.plot(range(1, epochs+1), test_loss_history, label='Test loss')
        plt.title("Loss over rounds")
        plt.xlabel("Epochs")
        plt.ylabel(loss)
        plt.show()

        plt.figure()
        plt.plot(range(1, epochs+1), train_acc_history, label='Train accuracy')
        plt.plot(range(1, epochs+1), test_acc_history, label='Test accuracy')
        plt.title("Accuracy over rounds")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()
