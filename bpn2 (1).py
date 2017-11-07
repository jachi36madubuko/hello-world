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
class bpn():
    def main(self):
        epochs = 3000
        batchSize = 128
        activation = 'sigmoid'
        learning_rate = 0.001
        hidden_neurons = 8


        # XOR  batch_size 1
        self.X = np.array([[0,0], [0,1], [1,0], [1,1]])
        self.Y = np.array([ [0],   [1],   [1],   [0]])


        # Breast cancer - batch_size 10
        #training_data_df = pd.read_csv("breast_cancer_wisconsin.csv")

        # Contraception - batch_size 10
        #training_data_df = pd.read_csv("contraceptive_method.csv")

        # Credit card default - batch_size 250
        #training_data_df = pd.read_csv("credit_card_default.csv")

        # Ionosphere - batch_size 128
        #training_data_df = pd.read_csv("credit_card_default.csv")

         #Wine - batch_size 128
        training_data_df = pd.read_csv("combined_red_white.csv")


        # Data scaling, create X and Y
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_training = scaler.fit_transform(training_data_df)
        scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
        self.X = scaled_training_df.drop('class', axis=1).values
        self.Y = scaled_training_df[['class']].values

        minx, maxx = 0, 1
        miny, maxy = 0, 1
        numx = int(maxx * 5 + 1)
        inputLayerSize, hiddenLayerSize, outputLayerSize = self.X.shape[1]+1, hidden_neurons, 1 #add +1 for bias

        funcs = {'sigmoid':  (lambda x: 1/(1 + np.exp(-x)),
                              lambda x: x * (1 - x),  (0,  1), .45),
                }
        (activate, activatePrime, (mina, maxa), L) = funcs[activation]

        # add a bias unit to the input layer
        self.X = np.concatenate((np.atleast_2d(np.ones(self.X.shape[0])).T, self.X), axis=1)

        # Random initial weights
        r0 = math.sqrt(2.0/(inputLayerSize))
        r1 = math.sqrt(2.0/(hiddenLayerSize))
        self.Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize),low=-r0,high=r0)
        self.Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize),low=-r1,high=r1)

        def next_batch(X, Y):
            for i in np.arange(0, X.shape[0], batchSize):
                yield (X[i:i + batchSize], Y[i:i + batchSize])

        start = time.time()
        lossHistory = []
        for i in range(epochs):                         # Training:
            epochLoss = []

            for (Xb, Yb) in next_batch(self.X, self.Y):

                H = activate(np.dot(Xb, self.Wh))            # hidden layer results
                Z = activate(np.dot(H,  self.Wz))            # output layer results
                E = Yb - Z                              # how much we missed (error)
                epochLoss.append(np.sum(E**2))

                dZ = E * activatePrime(Z)               # delta Z
                dH = dZ.dot(self.Wh.T) * activatePrime(H)    # delta H
                self.Wh += H.T.dot(dZ) * learning_rate       # update output layer weights
                self.Wz += Xb.T.dot(dH) * learning_rate      # update hidden layer weights

            mse = np.average(epochLoss)
            lossHistory.append(mse)

            self.X[:, 1] += maxx/(numx-1)/2
            H = activate(np.dot(self.X, self.Wh))
            Z = activate(np.dot(H, self.Wz))
            Z = ((miny - maxy) * Z - maxa * miny + maxy * mina)/(mina - maxa)

            end = time.time()

            plt.plot(lossHistory)
            plt.show()



        print('[', inputLayerSize, hiddenLayerSize, outputLayerSize, ']',
              'Activation:', activation, 'Iterations:', epochs,
              'Learning rate:', learning_rate, 'Final loss:', mse, 'Time:', end - start)

        return inputLayerSize*hiddenLayerSize +  hiddenLayerSize*outputLayerSize

    def test(self,WM):
        split = self.Wh.size
        HWM = np.reshape(WM[:split],(split,1))
        OWM = np.reshape(WM[split:],(WM.size-split,1))
        self.Wh = np.reshape(HWM,self.Wh.shape)
        self.Wz = np.reshape(OWM,self.Wz.shape)

        funcs = {'sigmoid':  (lambda x: 1/(1 + np.exp(-x)),
                      lambda x: x * (1 - x),  (0,  1), .45),
                }
        (activate, activatePrime, (mina, maxa), L) = funcs['sigmoid']


        #predict
#        self.X = np.array([[0,0], [0,1], [1,0], [1,1]])
#        self.Y = np.array([ [0],   [1],   [1],   [0]])
        H = activate(np.dot(self.X, self.Wh))            # hidden layer results
        Z = activate(np.dot(H,  self.Wz))            # output layer results
        E = Yb - Z                              # how much we missed (error)
        return np.average(np.sum(E**2))


b = bpn()
def main():
    return b.main()

def test(WM):
    return b.test(WM)
