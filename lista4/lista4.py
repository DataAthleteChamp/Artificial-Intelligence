import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

input_data = pd.read_csv('input_data.csv')
target_data = pd.read_csv('target_data.csv', usecols=[0, 1]) # 3 kolumna to same 0

#wymiary takie same
assert input_data.shape[0] == target_data.shape[0]


X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42) #random state przy każdym odpaleniu kodu taki sam podział

# reggresja i wartości ciagle

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))  #wyjście x y

    model.compile(optimizer='adam', loss='mean_squared_error') #adam adaptive nie wymaga dostrajania
    return model

# Adam jest algorytmem optymalizującym pewną zadaną funkcję kosztu opierającym się na stocha- stic gradient descent.
# Stochastic gradient descent różni się od czystego gradient descent tym,
# że nie optymalizuje funkcji dla wszystkich danych treningowych,
# tylko lokalnie dla kolejnych batchy danych.

model = build_model(X_train.shape[1])

#uczenie
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2) #100 rzy przejdzie przez dane po 32 przykady

y_pred = model.predict(X_test)

for i in range(5):
    print(f'Rzeczywiste: {y_test.iloc[i].values}, Przewidziane: {y_pred[i]}')

mse = mean_squared_error(y_test, y_pred)
print(f'Błąd średniokwadratowy: {mse}')


#k krotna walidacja krzyżowa  k=5
#walidacja jako proces po zakończonym treningu

def cross_validate_model(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

        model = build_model(X_train_kf.shape[1])
        model.fit(X_train_kf, y_train_kf, epochs=100, batch_size=32)

        y_pred_kf = model.predict(X_test_kf)
        mse_kf = mean_squared_error(y_test_kf, y_pred_kf)
        mse_scores.append(mse_kf)

    return mse_scores


scores = cross_validate_model(input_data, target_data)
print(f'Średni MSE z walidacji krzyżowej: {np.mean(scores)}')
