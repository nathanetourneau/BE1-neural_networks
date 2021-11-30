import numpy as np


def sigmoid(preactivation):
    return 1/(1+np.exp(-preactivation))


def sigmoid_backwards(preactivation, activation, cache):
    return activation*(1-activation)*cache


def relu(preactivation):
    return np.maximum(preactivation, 0)


def relu_backwards(preactivation, activation, cache):
    return cache * (activation > 0)


def linear(preactivation):
    return preactivation


def linear_backwards(preactivation, activation, cache):
    return cache


def tanh(preactivation):
    exps = np.exp(-2 * preactivation)
    return (1 - exps) / (1 + exps)


def tanh_backwards(preactivation, activation, cache):
    return (1 - activation**2) * cache
