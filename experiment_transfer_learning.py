from src.data import lecture_cifar, decoupage_donnees, one_hot_encoding, make_batches, descripteur_hog, cross_validation
from src.activations import sigmoid, sigmoid_backwards, relu, relu_backwards, linear, linear_backwards, tanh, tanh_backwards
from src.neuralnetwork import NeuralNetwork, NeuralNetworkTensorFlowLike, NeuralNetworkTwoHiddenLayers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    PATH = "../cifar-10-python/cifar-10-batches-py"
    NUM_CLASSES = 10

    # Load data and preprocess it
    x_raw, y = lecture_cifar(PATH, 5)
    y = one_hot_encoding(y, NUM_CLASSES)

    res = []

    for use_hog, use_transfer_learning in [(False, True), (True, False)]:
        if use_hog:
            print("Processing HOG features")
            x = descripteur_hog(x_raw)
            print("Done")

        if use_transfer_learning:
            x = x_raw.reshape(-1, 32, 32, 3)
            feature_extractor = tf.keras.models.Sequential()
            feature_extractor = tf.keras.models.Sequential()
            vgg19 = tf.keras.applications.VGG19(
                include_top=False, input_shape=(32, 32, 3))
            feature_extractor.add(vgg19)
            feature_extractor.add(tf.keras.layers.Flatten())

            x = feature_extractor.predict(x)

        x_train, y_train, x_test, y_test = decoupage_donnees(x, y, 0.8)

        activation = 'relu'
        epochs = 150
        batch_size = 64
        learning_rate = 1e-3
        loss = 'Cross-entropy'

        min_val = np.min(x_train)
        max_val = np.max(x_train)

        reg_coef = 0.1

        accuracies = {}

        activation_fn = relu
        activation_fn_backwards = relu_backwards

        # Standardization to [0, 1] : greater than 0, better for the relu
        x_train = (x_train - min_val)/(max_val - min_val)
        x_test = (x_test - min_val)/(max_val - min_val)

        m = NeuralNetwork(activation_fn, activation_fn_backwards,
                          loss, units=512, verbose=True)
        metrics_history = m.train(x_train, y_train, x_test, y_test, epochs=epochs,
                                  batch_size=batch_size, learning_rate=learning_rate, reg_coef=reg_coef)
        train_loss_history, train_acc_history, test_loss_history, test_acc_history = metrics_history

        res.append(test_acc_history)

    plt.figure()
    plt.plot(range(1, epochs+1), res[0], label='HOG')
    plt.plot(range(1, epochs+1), res[1], label='VGG-19')
    plt.title("Accuracy over rounds")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
