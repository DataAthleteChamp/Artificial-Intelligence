import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt


def sigmoid(x, derivative=False):
    sigm = 1 / (1 + np.exp(-x))
    if derivative:
        return sigm * (1 - sigm)
    return sigm


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1.0, 0.0)
    return np.maximum(0, x)


def initialize_weights_xavier(input_size, hidden_size, output_size):
    limit_hidden = sqrt(6 / (input_size + hidden_size))
    w_hidden = np.random.uniform(-limit_hidden, limit_hidden, (input_size + 1, hidden_size))

    limit_output = sqrt(6 / (hidden_size + output_size))
    w_output = np.random.uniform(-limit_output, limit_output, (hidden_size + 1, output_size))

    return w_hidden, w_output
# weight = U [-(1/sqrt(n)), 1/sqrt(n)]


def initialize_weights():
    w_hidden = np.random.uniform(-1, 1, (3, 2)) #3 neurony 2 wejscia 1 bias
    w_output = np.random.uniform(-1, 1, 3)
    return w_hidden, w_output


def backward_propagation(X, y, w_hidden, w_output, learning_rate, momentum, activation_func, max_epochs, min_error):
    # Dodanie biasu do danych wejściowych
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    errors = []
    w_hidden_update = np.zeros_like(w_hidden)
    w_output_update = np.zeros_like(w_output)

    for epoch in range(max_epochs):
        # FORWARD PROPAGATION
        # Obliczanie aktywacji dla warstwy ukrytej (w tym bias)
        hidden_layer_input = np.dot(X_bias, w_hidden)
        hidden_layer_output = activation_func(hidden_layer_input)
        hidden_layer_output_bias = np.hstack([hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))])

        # Obliczanie aktywacji dla warstwy wyjściowej
        output_layer_input = np.dot(hidden_layer_output_bias, w_output)
        predicted_output = activation_func(output_layer_input)

        # Obliczanie błędu wyjścia
        error = y - predicted_output

        # BACKWARD PROPAGATION
        # Obliczanie pochodnej błędu dla wyjścia
        d_predicted_output = error * activation_func(predicted_output, derivative=True)

        # Przekształcenie dla użycia w dalszych obliczeniach
        d_predicted_output_reshaped = d_predicted_output.reshape(-1, 1)

        # Propagacja błędu do warstwy ukrytej i obliczanie pochodnej
        error_hidden_layer = np.dot(d_predicted_output_reshaped, w_output[:-1].reshape(1, -1))
        d_hidden_layer = error_hidden_layer * activation_func(hidden_layer_output, derivative=True)

        # Aktualizacja wag dla warstwy wyjściowej
        w_output_update_new = learning_rate * np.dot(hidden_layer_output_bias.T, d_predicted_output_reshaped).flatten() + momentum * w_output_update

        # Aktualizacja wag dla warstwy ukrytej
        w_hidden_update_new = learning_rate * np.dot(X_bias.T, d_hidden_layer).reshape(w_hidden.shape) + momentum * w_hidden_update

        # Zastosowanie aktualizacji wag
        w_output += w_output_update_new
        w_hidden += w_hidden_update_new

        # Zapamiętanie poprzednich aktualizacji dla pędu
        w_hidden_update = w_hidden_update_new
        w_output_update = w_output_update_new

        # Obliczanie bieżącego błędu dla monitorowania
        current_error = np.mean(np.abs(error))
        errors.append(current_error)

        # Warunek zakończenia treningu
        if current_error < min_error:
            break

    return w_hidden, w_output, errors


momentum = 0.95
max_epochs = 10000
min_error = 0.05
learning_rate = 0.1

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])


def test_network(X, w_hidden, w_output, activation_func):
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    hidden_layer_input = np.dot(X_bias, w_hidden)
    hidden_layer_output = activation_func(hidden_layer_input)
    hidden_layer_output_bias = np.hstack([hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))])
    output_layer_input = np.dot(hidden_layer_output_bias, w_output)
    predicted_output = activation_func(output_layer_input)

    return predicted_output


for activation_func in [sigmoid, tanh, relu]:
    print(f"Trenowanie z funkcją aktywacji {activation_func.__name__}")
    w_hidden, w_output = initialize_weights()

    start_time = time.time()

    w_hidden, w_output, errors = backward_propagation(X, y, w_hidden, w_output, learning_rate, momentum,
                                                      activation_func, max_epochs, min_error)

    elapsed_time = time.time() - start_time
    final_error = errors[-1]

    plt.plot(errors)
    plt.xlabel('Epoki')
    plt.ylabel('Błąd średniokwadratowy')
    plt.title(f'Proces uczenia sieci neuronowej z funkcją aktywacji {activation_func.__name__}')
    plt.savefig(f'Proces uczenia sieci neuronowej z funkcją aktywacji {activation_func.__name__}.jpg')
    plt.show()
    plt.close()

    test_results = test_network(X, w_hidden, w_output, activation_func)
    print(f"Wyniki testu dla funkcji aktywacji {activation_func.__name__}:")
    for input_data, predicted in zip(X, test_results):
        print(f"Input: {input_data} -> Predicted: {predicted}")

    print(f"Czas wykonania dla {activation_func.__name__}: {elapsed_time:.2f} sekund")
    print(f"Błąd na końcu treningu dla {activation_func.__name__}: {final_error:.4f}")

# Możesz spróbować innych metod inicjalizacji wag, np. inicjalizacji Xavier'a lub He.
# https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
