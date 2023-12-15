import numpy as np
import matplotlib.pyplot as plt
import time

#prosty algorytm siatkowy

def sigmoid(x, derivative=False):
    sigm = 1 / (1 + np.exp(-x))
    if derivative:
        return sigm * (1 - sigm)
    return sigm

def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x)**2
    return np.tanh(x)

def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1.0, 0.0)
    return np.maximum(0, x)


def initialize_weights():
    w_hidden = np.random.uniform(-1, 1, (3, 2))
    w_output = np.random.uniform(-1, 1, 3)
    return w_hidden, w_output



def backward_propagation(X, y, w_hidden, w_output, learning_rate, momentum, activation_func, max_epochs, min_error):
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    errors = []
    w_hidden_update = np.zeros_like(w_hidden)
    w_output_update = np.zeros_like(w_output)

    for epoch in range(max_epochs):
        hidden_layer_input = np.dot(X_bias, w_hidden)
        hidden_layer_output = activation_func(hidden_layer_input)
        hidden_layer_output_bias = np.hstack([hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))])
        output_layer_input = np.dot(hidden_layer_output_bias, w_output)
        predicted_output = activation_func(output_layer_input)

        error = y - predicted_output
        d_predicted_output = error * activation_func(predicted_output, derivative=True)

        d_predicted_output_reshaped = d_predicted_output.reshape(-1, 1)
        error_hidden_layer = np.dot(d_predicted_output_reshaped, w_output[:-1].reshape(1, -1))
        d_hidden_layer = error_hidden_layer * activation_func(hidden_layer_output, derivative=True)

        w_output_update_new = learning_rate * np.dot(hidden_layer_output_bias.T,
                                                     d_predicted_output_reshaped).flatten() + momentum * w_output_update
        w_hidden_update_new = learning_rate * np.dot(X_bias.T, d_hidden_layer).reshape(
            w_hidden.shape) + momentum * w_hidden_update

        w_output += w_output_update_new
        w_hidden += w_hidden_update_new

        w_hidden_update = w_hidden_update_new
        w_output_update = w_output_update_new

        current_error = np.mean(np.abs(error))
        errors.append(current_error)

        if current_error < min_error:
            break

    return w_hidden, w_output, errors


def test_network(X, w_hidden, w_output, activation_func):
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    hidden_layer_input = np.dot(X_bias, w_hidden)
    hidden_layer_output = activation_func(hidden_layer_input)
    hidden_layer_output_bias = np.hstack([hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))])
    output_layer_input = np.dot(hidden_layer_output_bias, w_output)
    predicted_output = activation_func(output_layer_input)

    return predicted_output



learning_rates = [0.01, 0.1, 0.5, 0.3]
momentums = [0.9, 0.95, 0.98, 0.99]
activation_functions = [sigmoid, tanh, relu]
epochs_options = [1000, 3000, 5000, 7000, 10000]
min_error = 0.04

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])


best_error = float('inf')
best_params = {}

for lr in learning_rates:
    for momentum in momentums:
        for epochs in epochs_options:
            for activation_func in activation_functions:
                w_hidden, w_output = initialize_weights()
                _, _, errors = backward_propagation(X, y, w_hidden, w_output, lr, momentum, activation_func, epochs,
                                                    min_error)
                final_error = errors[-1] if errors else float('inf')

                if final_error < best_error:
                    best_error = final_error
                    best_params = {'learning_rate': lr, 'momentum': momentum, 'epochs': epochs,
                                   'activation_func': activation_func.__name__}

print("Najlepsze parametry:", best_params)
