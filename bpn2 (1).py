#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:57:26 2017

@author: gsiverts
"""

import numpy as np
import matplotlib.pyplot as plt
import math, time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

epochs = 3000
batchSize = 128
activation = 'sigmoid'
learning_rate = 0.001
hidden_neurons = 8


# XOR  batch_size 1
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])


# Breast cancer - batch_size 10
#training_data_df = pd.read_csv("breast_cancer_wisconsin.csv")

# Contraception - batch_size 10
#training_data_df = pd.read_csv("contraceptive_method.csv")

# Credit card default - batch_size 250
#training_data_df = pd.read_csv("credit_card_default.csv")

# Ionosphere - batch_size 128
#training_data_df = pd.read_csv("credit_card_default.csv")

# Wine - batch_size 128
training_data_df = pd.read_csv("combined_red_white.csv")


# Data scaling, create X and Y
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training = scaler.fit_transform(training_data_df)
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
X = scaled_training_df.drop('class', axis=1).values
Y = scaled_training_df[['class']].values

minx, maxx = 0, 1
miny, maxy = 0, 1
numx = int(maxx * 5 + 1)
inputLayerSize, hiddenLayerSize, outputLayerSize = X.shape[1]+1, hidden_neurons, 1 #add +1 for bias

funcs = {'sigmoid':  (lambda x: 1/(1 + np.exp(-x)),
                      lambda x: x * (1 - x),  (0,  1), .45),
        }
(activate, activatePrime, (mina, maxa), L) = funcs[activation]

# add a bias unit to the input layer
X = np.concatenate((np.atleast_2d(np.ones(X.shape[0])).T, X), axis=1)

# Random initial weights
r0 = math.sqrt(2.0/(inputLayerSize))
r1 = math.sqrt(2.0/(hiddenLayerSize))
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize),low=-r0,high=r0)
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize),low=-r1,high=r1)

def next_batch(X, Y):
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], Y[i:i + batchSize])

start = time.time()
lossHistory = []

for i in range(epochs):                         # Training:
    epochLoss = []

    for (Xb, Yb) in next_batch(X, Y):

        H = activate(np.dot(Xb, Wh))            # hidden layer results
        Z = activate(np.dot(H,  Wz))            # output layer results
        E = Yb - Z                              # how much we missed (error)
        epochLoss.append(np.sum(E**2))

        dZ = E * activatePrime(Z)               # delta Z
        dH = dZ.dot(Wz.T) * activatePrime(H)    # delta H
        Wz += H.T.dot(dZ) * learning_rate       # update output layer weights
        Wh += Xb.T.dot(dH) * learning_rate      # update hidden layer weights

    mse = np.average(epochLoss)
    lossHistory.append(mse)

X[:, 1] += maxx/(numx-1)/2
H = activate(np.dot(X, Wh))
Z = activate(np.dot(H, Wz))
Z = ((miny - maxy) * Z - maxa * miny + maxy * mina)/(mina - maxa)

end = time.time()

plt.plot(lossHistory)
plt.show()

print('[', inputLayerSize, hiddenLayerSize, outputLayerSize, ']',
      'Activation:', activation, 'Iterations:', epochs,
      'Learning rate:', learning_rate, 'Final loss:', mse, 'Time:', end - start)