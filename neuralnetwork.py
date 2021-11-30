import numpy as np
from data import make_batches


class NeuralNetworkTwoHiddenLayers:
    def __init__(self, activation_fn, activation_fn_backwards, loss, units1=64, units2=64, verbose=True, dropout=0.5):
        self.weights = {}
        self.activation_fn = activation_fn
        self.activation_fn_backwards = activation_fn_backwards
        self.loss = loss
        self.verbose = verbose
        self.units1 = units1
        self.units2 = units2
        self.dropout = dropout

    def batch_step(self, X_train, Y_train, learning_rate=1e-4, reg_coef=0.1):
        ##########################
        # Génération des données #
        ##########################
        N = len(Y_train)

        loss_values = []

        W1, b1, W2, b2, W3, b3 = (
            self.weights["W1"],
            self.weights["b1"],
            self.weights["W2"],
            self.weights["b2"],
            self.weights["W3"],
            self.weights["b3"]
        )
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        I1 = X_train.dot(W1) + b1  # Potentiel d'entrée de la couche cachée
        # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
        O1 = self.activation_fn(I1)

        mask1 = np.random.binomial(1, self.dropout, O1.shape) / self.dropout

        O1 = mask1 * O1
        # O1 = np.maximum(I1, 0)
        I2 = O1.dot(W2) + b2  # Potentiel d'entrée de la couche de sortie
        # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
        O2 = self.activation_fn(I2)

        mask2 = np.random.binomial(1, self.dropout, O2.shape) / self.dropout

        O2 = mask2 * O2

        I3 = O2.dot(W3) + b3

        O3 = self.activation_fn(I3)

        mask3 = np.random.binomial(1, self.dropout, O3.shape) / self.dropout

        O3 = mask3 * O3

        if self.loss == 'MSE':
            Y_pred = O3  # Les valeurs prédites sont les sorties de la couche de sortie
            loss = 0.5*np.square(Y_pred - Y_train).sum()/N

        elif self.loss == 'Cross-entropy':
            logits = I3
            softmax_probs = np.exp(
                logits-np.max(logits, axis=1, keepdims=True))
            Y_pred = softmax_probs/softmax_probs.sum(axis=1, keepdims=True)
            loss = - np.diag(Y_train@(np.log(Y_pred).T)).sum()/N
        else:
            raise ValueError(
                "Undefined loss, valid entries are 'MSE' and 'Cross-entropy'")

        ########################################################
        # Calcul et affichage de la fonction perte de type MSE #
        ########################################################

        loss_values.append(loss)

        # In both cases, same gradient wrt O2
        delta_O3 = Y_pred - Y_train  # N*D_out

        delta_O3 *= mask3

        if self.loss == 'Cross-entropy':
            delta_I3 = delta_O3
        elif self.loss == 'MSE':
            delta_I3 = self.activation_fn_backwards(
                I3, O3, delta_O3)  # N*D_out
        else:
            raise ValueError(
                "Undefined loss, valid entries are 'MSE' and 'Cross-entropy'")

        dW3 = (O2.T).dot(delta_I3)  # D_h * D_out
        db3 = np.sum(delta_I3, axis=0)  # 1*D_out

        delta_O2 = delta_I3.dot(W3.T)  # N*D_h
        delta_O2 *= mask2
        delta_I2 = self.activation_fn_backwards(I2, O2, delta_O2)  # N*D_h

        dW2 = (O1.T).dot(delta_I2)  # D_h * D_out
        db2 = np.sum(delta_I2, axis=0)  # 1*D_out

        delta_O1 = delta_I2.dot(W2.T)  # N*D_h
        delta_O1 *= mask1
        delta_I1 = self.activation_fn_backwards(I1, O1, delta_O1)  # N*D_h
        dW1 = (X_train.T).dot(delta_I1)
        db1 = np.sum(delta_I1, axis=0)

        W3 -= learning_rate * (dW3 + reg_coef * W3)
        b3 -= learning_rate * (db3 + reg_coef * b3)
        W2 -= learning_rate * (dW2 + reg_coef * W2)
        b2 -= learning_rate * (db2 + reg_coef * b2)
        W1 -= learning_rate * (dW1 + reg_coef * W1)
        b1 -= learning_rate * (db1 + reg_coef * b1)

        (
            self.weights["W1"],
            self.weights["b1"],
            self.weights["W2"],
            self.weights["b2"],
            self.weights["W3"],
            self.weights["b3"],
        ) = (W1, b1, W2, b2, W3, b3)
        return loss

    def train_one_epoch(self, x_train, y_train, batch_size=64, learning_rate=1e-4, reg_coef=0.1):
        train_loss = 0
        batch_count = 0

        for x_batch, y_batch in make_batches(x_train, y_train, batch_size):
            train_loss += self.batch_step(x_batch, y_batch,
                                          learning_rate, reg_coef)
            batch_count += 1

        train_loss /= batch_count
        return train_loss

    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=10, batch_size=64, learning_rate=1e-2, reg_coef=0.1):
        D_in = x_train.shape[1]
        D_h1 = self.units1  # D_h le nombre de neurones de la couche cachée
        D_h2 = self.units2
        # D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
        D_out = 10

        self.weights["W1"] = np.sqrt(
            2/D_in) * (2*np.random.random((D_in, D_h1)) - 1)
        self.weights["b1"] = 0.01 * np.ones((1, D_h1))
        self.weights["W2"] = np.sqrt(
            2/D_h1) * (2*np.random.random((D_h1, D_h2)) - 1)
        self.weights["b2"] = 0.01 * np.ones((1, D_h2))
        self.weights["W3"] = np.sqrt(
            2/D_h2) * (2*np.random.random((D_h2, D_out)) - 1)
        self.weights["b3"] = 0.01 * np.ones((1, D_out))

        train_loss_history = []
        train_acc_history = []

        if x_test is not None and y_test is not None:
            test_loss_history = []
            test_acc_history = []

        for epoch in range(epochs):
            if self.verbose:
                print(f'\nEpoch n°{epoch+1}/{epochs}')
                print('===========================================================')
            train_loss = self.train_one_epoch(
                x_train, y_train, batch_size, learning_rate, reg_coef)
            train_loss_history.append(train_loss)
            if self.verbose:
                print(f"Train loss : {train_loss:.3f}")

            y_pred = self.predict(x_train)
            train_accuracy = np.equal(np.argmax(y_pred, axis=1),
                                      np.argmax(y_train, axis=1)).sum()/len(y_train)
            train_acc_history.append(train_accuracy)
            if self.verbose:
                print(f"Train accuracy : {train_accuracy:.3f}")

            if x_test is not None and y_test is not None:
                test_loss = self.evaluate(x_test, y_test)
                test_loss_history.append(test_loss)
                if self.verbose:
                    print(f"Test loss : {test_loss:.3f}")

                y_pred = self.predict(x_test)
                test_accuracy = np.equal(np.argmax(y_pred, axis=1),
                                         np.argmax(y_test, axis=1)).sum()/len(y_test)
                test_acc_history.append(test_accuracy)
                if self.verbose:
                    print(f"Test accuracy : {test_accuracy :.3f}")

        if x_test is not None and y_test is not None:
            return train_loss_history, train_acc_history, test_loss_history, test_acc_history
        return train_loss_history, train_acc_history

    def predict(self, X):
        W1, b1, W2, b2, W3, b3 = (
            self.weights["W1"],
            self.weights["b1"],
            self.weights["W2"],
            self.weights["b2"],
            self.weights["W3"],
            self.weights["b3"]
        )

        I1 = X.dot(W1) + b1  # Potentiel d'entrée de la couche cachée
        # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
        O1 = self.activation_fn(I1)
        I2 = O1.dot(W2) + b2  # Potentiel d'entrée de la couche de sortie
        # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
        O2 = self.activation_fn(I2)

        I3 = O2.dot(W3) + b3
        O3 = self.activation_fn(I3)

        if self.loss == 'MSE':
            Y_pred = O3  # Les valeurs prédites sont les sorties de la couche de sortie
        elif self.loss == 'Cross-entropy':
            logits = I3
            softmax_probs = np.exp(
                logits-np.max(logits, axis=1, keepdims=True))
            Y_pred = softmax_probs/softmax_probs.sum(axis=1, keepdims=True)

        return Y_pred

    def evaluate(self, X, Y):
        N = len(X)
        Y_pred = self.predict(X)
        if self.loss == 'MSE':
            score = 0.5*np.square(Y_pred - Y).sum()/N
        elif self.loss == 'Cross-entropy':
            score = - np.diag(Y@(np.log(Y_pred).T)).sum()/N
        return score


class NeuralNetwork:
    def __init__(self, activation_fn, activation_fn_backwards, loss, units=64, verbose=True):
        self.weights = {}
        self.activation_fn = activation_fn
        self.activation_fn_backwards = activation_fn_backwards
        self.loss = loss
        self.verbose = verbose
        self.units = units

    def batch_step(self, X_train, Y_train, learning_rate=1e-4, reg_coef=0.1):
        ##########################
        # Génération des données #
        ##########################
        N = len(Y_train)

        loss_values = []

        W1, b1, W2, b2 = (
            self.weights["W1"],
            self.weights["b1"],
            self.weights["W2"],
            self.weights["b2"],
        )
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        I1 = X_train.dot(W1) + b1  # Potentiel d'entrée de la couche cachée
        # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
        O1 = self.activation_fn(I1)
        # O1 = np.maximum(I1, 0)
        I2 = O1.dot(W2) + b2  # Potentiel d'entrée de la couche de sortie
        # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
        O2 = self.activation_fn(I2)
        # O2 = np.maximum(I2, 0)

        if self.loss == 'MSE':
            Y_pred = O2  # Les valeurs prédites sont les sorties de la couche de sortie
            loss = 0.5*np.square(Y_pred - Y_train).sum()/N

        elif self.loss == 'Cross-entropy':
            logits = I2
            softmax_probs = np.exp(
                logits-np.max(logits, axis=1, keepdims=True))
            Y_pred = softmax_probs/softmax_probs.sum(axis=1, keepdims=True)
            loss = - np.diag(Y_train@(np.log(Y_pred).T)).sum()/N
        else:
            raise ValueError(
                "Undefined loss, valid entries are 'MSE' and 'Cross-entropy'")

        ########################################################
        # Calcul et affichage de la fonction perte de type MSE #
        ########################################################

        loss_values.append(loss)

        # In both cases, same gradient wrt O2
        delta_O2 = Y_pred - Y_train  # N*D_out
        if self.loss == 'Cross-entropy':
            delta_I2 = delta_O2
        elif self.loss == 'MSE':
            delta_I2 = self.activation_fn_backwards(
                I2, O2, delta_O2)  # N*D_out
        else:
            raise ValueError(
                "Undefined loss, valid entries are 'MSE' and 'Cross-entropy'")
        dW2 = (O1.T).dot(delta_I2)  # D_h * D_out
        db2 = np.sum(delta_I2, axis=0)  # 1*D_out

        delta_O1 = delta_I2.dot(W2.T)  # N*D_h
        delta_I1 = self.activation_fn_backwards(I1, O1, delta_O1)  # N*D_h
        dW1 = (X_train.T).dot(delta_I1)
        db1 = np.sum(delta_I1, axis=0)

        W2 -= learning_rate * (dW2 + reg_coef * W2)
        b2 -= learning_rate * (db2 + reg_coef * b2)
        W1 -= learning_rate * (dW1 + reg_coef * W1)
        b1 -= learning_rate * (db1 + reg_coef * b1)

        (
            self.weights["W1"],
            self.weights["b1"],
            self.weights["W2"],
            self.weights["b2"],
        ) = (W1, b1, W2, b2)
        return loss

    def train_one_epoch(self, x_train, y_train, batch_size=64, learning_rate=1e-4, reg_coef=0.1):
        train_loss = 0
        batch_count = 0

        for x_batch, y_batch in make_batches(x_train, y_train, batch_size):
            train_loss += self.batch_step(x_batch, y_batch,
                                          learning_rate, reg_coef)
            batch_count += 1

        train_loss /= batch_count
        return train_loss

    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=10, batch_size=64, learning_rate=1e-2, reg_coef=0.1):
        D_in = x_train.shape[1]
        D_h = self.units  # D_h le nombre de neurones de la couche cachée
        # D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
        D_out = 10

        self.weights["W1"] = 0.01 * (2*np.random.random((D_in, D_h)) - 1)
        self.weights["b1"] = 0.01 * np.ones((1, D_h))
        self.weights["W2"] = 0.01 * (2*np.random.random((D_h, D_out)) - 1)
        self.weights["b2"] = 0.01 * np.ones((1, D_out))

        train_loss_history = []
        train_acc_history = []

        if x_test is not None and y_test is not None:
            test_loss_history = []
            test_acc_history = []

        for epoch in range(epochs):
            if self.verbose:
                print(f'\nEpoch n°{epoch+1}/{epochs}')
                print('===========================================================')
            train_loss = self.train_one_epoch(
                x_train, y_train, batch_size, learning_rate, reg_coef)
            train_loss_history.append(train_loss)
            if self.verbose:
                print(f"Train loss : {train_loss:.3f}")

            y_pred = self.predict(x_train)
            train_accuracy = np.equal(np.argmax(y_pred, axis=1),
                                      np.argmax(y_train, axis=1)).sum()/len(y_train)
            train_acc_history.append(train_accuracy)
            if self.verbose:
                print(f"Train accuracy : {train_accuracy:.3f}")

            if x_test is not None and y_test is not None:
                test_loss = self.evaluate(x_test, y_test)
                test_loss_history.append(test_loss)
                if self.verbose:
                    print(f"Test loss : {test_loss:.3f}")

                y_pred = self.predict(x_test)
                test_accuracy = np.equal(np.argmax(y_pred, axis=1),
                                         np.argmax(y_test, axis=1)).sum()/len(y_test)
                test_acc_history.append(test_accuracy)
                if self.verbose:
                    print(f"Test accuracy : {test_accuracy :.3f}")

        if x_test is not None and y_test is not None:
            return train_loss_history, train_acc_history, test_loss_history, test_acc_history
        return train_loss_history, train_acc_history

    def predict(self, X):
        W1, b1, W2, b2 = (
            self.weights["W1"],
            self.weights["b1"],
            self.weights["W2"],
            self.weights["b2"],
        )

        I1 = X.dot(W1) + b1  # Potentiel d'entrée de la couche cachée
        # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
        O1 = self.activation_fn(I1)
        I2 = O1.dot(W2) + b2  # Potentiel d'entrée de la couche de sortie
        # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
        O2 = self.activation_fn(I2)

        if self.loss == 'MSE':
            Y_pred = O2  # Les valeurs prédites sont les sorties de la couche de sortie

        elif self.loss == 'Cross-entropy':
            logits = I2
            softmax_probs = np.exp(
                logits-np.max(logits, axis=1, keepdims=True))
            Y_pred = softmax_probs/softmax_probs.sum(axis=1, keepdims=True)

        return Y_pred

    def evaluate(self, X, Y):
        N = len(X)
        Y_pred = self.predict(X)
        if self.loss == 'MSE':
            score = 0.5*np.square(Y_pred - Y).sum()/N
        elif self.loss == 'Cross-entropy':
            score = - np.diag(Y@(np.log(Y_pred).T)).sum()/N
        return score


class NeuralNetworkTensorFlowLike:

    def __init__(self, units_list, activation_fn_list, activation_fn_backwards_list, loss):
        self.weights = [None for _ in units_list]
        self.bias = [None for _ in units_list]
        self.weights_gradients = [None for _ in units_list]
        self.bias_gradients = [None for _ in units_list]
        self.preactivations = [None for _ in units_list]
        self.activations = [None for _ in units_list]
        self.preactivations_grads = [None for _ in units_list]
        self.activations_grads = [None for _ in units_list]

        self.units_list = units_list
        self.activation_fn_list = activation_fn_list
        self.activation_fn_backwards_list = activation_fn_backwards_list
        self.loss = loss

    def train(self, x_train, y_train, x_test=None, y_test=None, epochs=10, batch_size=64, learning_rate=1e-2, reg_coef=0.1):
        D_in = x_train.shape[1]

        for index, units in enumerate(self.units_list):
            previous_units = D_in if index == 0 else self.units_list[index-1]

            self.weights[index] = 0.01 * \
                (2*np.random.random((previous_units, units)) - 1)
            self.bias[index] = 0.01 * np.ones((1, units))

        train_loss_history = []
        train_acc_history = []

        if x_test is not None and y_test is not None:
            test_loss_history = []
            test_acc_history = []

        for epoch in range(epochs):
            print(f'\nEpoch n°{epoch+1}/{epochs}')
            print('===========================================================')
            train_loss = self.train_one_epoch(
                x_train, y_train, batch_size, learning_rate, reg_coef)
            train_loss_history.append(train_loss)
            print(f"Train loss : {train_loss:.3f}")

            y_pred = self.predict(x_train)
            train_accuracy = np.equal(np.argmax(y_pred, axis=1),
                                      np.argmax(y_train, axis=1)).sum()/len(y_train)
            train_acc_history.append(train_accuracy)
            print(f"Train accuracy : {train_accuracy:.3f}")

            if x_test is not None and y_test is not None:
                test_loss = self.evaluate(x_test, y_test)
                test_loss_history.append(test_loss)
                print(f"Test loss : {test_loss:.3f}")

                y_pred = self.predict(x_test)
                test_accuracy = np.equal(np.argmax(y_pred, axis=1),
                                         np.argmax(y_test, axis=1)).sum()/len(y_test)
                test_acc_history.append(test_accuracy)
                print(f"Test accuracy : {test_accuracy :.3f}")

        if x_test is not None and y_test is not None:
            return train_loss_history, train_acc_history, test_loss_history, test_acc_history
        return train_loss_history, train_acc_history

    def train_one_epoch(self, x_train, y_train, batch_size=64, learning_rate=1e-4, reg_coef=0.1):
        train_loss = 0
        batch_count = 0

        for x_batch, y_batch in make_batches(x_train, y_train, batch_size):
            train_loss += self.batch_step(x_batch, y_batch,
                                          learning_rate, reg_coef)
            batch_count += 1

        train_loss /= batch_count
        return train_loss

    def predict(self, X):
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        nb_layers = len(self.units_list)
        for index in range(nb_layers):
            activation_fn = self.activation_fn_list[index]

            Wi = self.weights[index]
            bi = self.bias[index]

            if index > 0:
                preactivation = (self.activations[index-1]).dot(Wi) + bi
            else:
                preactivation = X.dot(Wi) + bi

            self.preactivations[index] = preactivation
            self.activations[index] = activation_fn(preactivation)

        if self.loss == 'MSE':
            # Les valeurs prédites sont les sorties de la couche de sortie
            Y_pred = self.activations[nb_layers-1]

        elif self.loss == 'Cross-entropy':
            logits = self.preactivations[nb_layers-1]
            softmax_probs = np.exp(
                logits-np.max(logits, axis=1, keepdims=True))
            Y_pred = softmax_probs/softmax_probs.sum(axis=1, keepdims=True)
        else:
            raise ValueError(
                "Undefined loss, valid entries are 'MSE' and 'Cross-entropy'")

        return Y_pred

    def evaluate(self, X, Y):
        N = len(X)
        Y_pred = self.predict(X)
        if self.loss == 'MSE':
            score = 0.5*np.square(Y_pred - Y).sum()/N
        elif self.loss == 'Cross-entropy':
            score = - np.diag(Y@(np.log(Y_pred).T)).sum()/N
        return score

    def batch_step(self, X_train, Y_train, learning_rate=1e-4, reg_coef=0.1):
        ##########################
        # Génération des données #
        ##########################
        nb_layers = len(self.units_list)
        N = len(Y_train)

        loss_values = []

        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        Y_pred = self.predict(X_train)

        if self.loss == 'MSE':
            loss = 0.5*np.square(Y_pred - Y_train).sum()/N

        elif self.loss == 'Cross-entropy':
            loss = -np.diag(Y_train@(np.log(Y_pred).T)).sum()/N

        loss_values.append(loss)

        ########################################################
        # Calcul et affichage de la fonction perte de type MSE #
        ########################################################

        # In both cases, same gradient wrt O2
        activation_fn_backwards = self.activation_fn_backwards_list[nb_layers-1]
        preactivation = self.preactivations[nb_layers-1]
        activation = self.activations[nb_layers-1]

        self.activations_grads[nb_layers-1] = Y_pred - Y_train  # N*D_out
        self.preactivations_grads[nb_layers-1] = activation_fn_backwards(
            preactivation, activation, self.activations_grads[nb_layers-1])  # N*D_out

        self.weights_gradients[nb_layers-1] = ((self.activations[nb_layers-2]).T).dot(
            self.preactivations_grads[nb_layers-1])
        self.bias_gradients[nb_layers -
                            1] = np.sum(self.preactivations_grads[nb_layers-1], axis=0)

        for index in range(nb_layers-2, -1, -1):
            activation_fn_backwards = self.activation_fn_backwards_list[index]
            preactivation = self.preactivations[index]
            activation = self.activations[index]

            self.activations_grads[index] = (
                self.preactivations[index+1]).dot((self.weights[index+1]).T)
            self.preactivations_grads[index] = activation_fn_backwards(
                preactivation, activation, self.activations_grads[index])

            if index > 0:
                self.weights_gradients[index] = (
                    (self.activations[index-1]).T).dot(self.preactivations_grads[index])
            else:
                self.weights_gradients[index] = ((X_train).T).dot(
                    self.preactivations_grads[index])

            self.bias_gradients[index] = np.sum(
                self.preactivations_grads[index], axis=0)

        for index in range(nb_layers):
            Wi = self.weights[index]
            dWi = self.weights_gradients[index]
            bi = self.bias[index]
            dbi = self.bias_gradients[index]

            self.weights[index] -= learning_rate * (dWi + reg_coef * Wi)
            self.bias[index] -= learning_rate * (dbi + reg_coef * bi)

        return loss
