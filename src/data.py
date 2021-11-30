import numpy as np
import pickle
import math

from skimage.feature import hog
from skimage.color import rgb2gray

path = "./cifar-10-python/cifar-10-batches-py"


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
    # Clés : [b'batch_label', b'labels', b'data', b'filenames']


def lecture_cifar(path: str, nb_batches):
    X = np.zeros(((nb_batches + 1) * 10000, 3072))
    Y = np.zeros(((nb_batches + 1) * 10000))
    for i in range(1, nb_batches + 1):
        batch_path = f"{path}/data_batch_{str(i)}"
        new_dict = unpickle(batch_path)
        batch_array = new_dict[b"data"]
        batch_labels = new_dict[b"labels"]
        X[(i - 1) * 10000: i * 10000, :] = batch_array
        Y[(i - 1) * 10000: i * 10000] = batch_labels

    new_dict = unpickle(f"{path}/test_batch")
    batch_array = new_dict[b"data"]
    batch_labels = new_dict[b"labels"]
    X[nb_batches * 10000: (nb_batches + 1) * 10000, :] = batch_array
    Y[nb_batches * 10000: (nb_batches + 1) * 10000] = batch_labels

    X = np.float32(X)
    Y = Y.astype(int)

    return X, Y


def decoupage_donnees(X, Y, ratio=0.8, small_sample=False):
    N = X.shape[0]
    indices = np.array(range(N))
    np.random.shuffle(indices)

    if small_sample:
        X_train = X[indices[:500], :]
        Y_train = Y[indices[:500]]

        X_test = X[indices[-100:], :]
        Y_test = Y[indices[-100:]]

        return X_train, Y_train, X_test, Y_test

    M = int(ratio * N)
    X_train = X[indices[:M], :]
    Y_train = Y[indices[:M]]

    X_test = X[indices[M:], :]
    Y_test = Y[indices[M:]]

    return X_train, Y_train, X_test, Y_test


def decoupe_pour_cross_validation(X, Y, nombre_folds):

    N = X.shape[0]
    indices = np.array(range(N))
    np.random.shuffle(indices)

    folds = []

    for fold in range(nombre_folds-1):
        M = math.floor(N/nombre_folds)
        X_fold = X[indices[fold*M:(fold+1)*M], :]
        Y_fold = Y[indices[fold*M:(fold+1)*M]]
        folds.append((X_fold, Y_fold))

    X_fold = X[indices[(nombre_folds-1)*M:], :]
    Y_fold = Y[indices[(nombre_folds-1)*M:]]

    folds.append((X_fold, Y_fold))

    return folds


def cross_validation(score_function, X, Y, nombre_folds):
    folds = decoupe_pour_cross_validation(X, Y, nombre_folds)
    train_errors = [None for _ in range(nombre_folds)]
    val_errors = [None for _ in range(nombre_folds)]

    for val_index in range(nombre_folds):
        print(f"*** Fold n°{val_index+1}/{nombre_folds} ***\n")
        x_data = np.vstack([folds[k][0]
                            for k in range(nombre_folds) if k != val_index])
        y_data = np.vstack([folds[k][1]
                            for k in range(nombre_folds) if k != val_index])
        x_val = folds[val_index][0]

        y_val = folds[val_index][1]

        train_error, val_error = score_function(x_data, y_data, x_val, y_val)

        train_errors[val_index] = train_error
        print(f"Train error : {train_error}")
        val_errors[val_index] = val_error
        print(f"Validation error : {val_error}")
        print('\n')

    return sum(train_errors)/len(train_errors), sum(val_errors)/len(val_errors)


def one_hot_encoding(y, nb_classes):
    return np.eye(nb_classes)[y]


def make_batches(x, y, batch_size):
    nb_batches = math.ceil(len(x)/batch_size)
    for batch_number in range(nb_batches-1):
        yield (
            x[batch_size*batch_number:batch_size*(batch_number+1), :],
            y[batch_size*batch_number:batch_size*(batch_number+1)]
        )
    yield x[batch_size*(nb_batches-1):, :], y[batch_size*(nb_batches-1):]


def descripteur_hog(X):
    """
    Transforme le format des images obtenues avec lecture_cifar, et utilise le descripteur HOG, et retransforme dans le format initial les images.
    Parameters
    ----------
    X : np.array()
        Images après utilisation du descripteur HOG

    Returns
    -------
    HOG : np.array()
        Images après utlisation du descripteur HOG
    """
    X_hog = []
    for i in range(len(X)):
        # Reshape les différents composants (rouge, bleu et vert) de l'image
        imr = np.reshape(X[i, :1024], (32, 32))
        imb = np.reshape(X[i, 1024:2048], (32, 32))
        img = np.reshape(X[i, 2048:], (32, 32))
        im = np.stack([imr, imb, img], axis=2)
        # Gray scale
        im = rgb2gray(im)
        # Utilisation du descripteur HOG
        im_hog = hog(im)
        # Reshape dans le bon format (1,1024)
        X_hog.append(im_hog)
    HOG = np.array(X_hog)
    return HOG
