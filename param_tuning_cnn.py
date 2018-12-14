# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:45:07 2018

@author: mgreen13
"""


import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optomizers import adam
import seaborn as sns
import matplotlib.pyplot as plt

x_train = np.load('xTrain.csv.npy')
x_test = np.load('xTest.csv.npy')
x_eval = np.load('xEval.csv.npy')

y_train = np.load('yTrain.csv.npy')
y_test = np.load('yTest.csv.npy')
y_eval = np.load('yEval.csv.npy')

learn_rate = [0.01,0.1]
dropout = [0.2,0.4,0.6]
num_filters = [16,32,64]
num_neurons = [32,64,128]
acuraccy = []

for l in learn_rate:
    adam_learn = adam()
    for d in dropout:
        for n in num_neurons:
            for f in num_filters:
                model = Sequential() 
                model.add(Conv2D(f, (3,3), activation='relu', input_shape=(28,28,4)))
                model.add(Conv2D(2*f, (3,3), activation='relu'))
                model.add(MaxPool2D(pool_size=(2,2)))
                # INCREASE DROPOUT WITH EACH LAYER
                model.add(Dropout(d))
                model.add(Conv2D(2*f, (3,3), activation='relu'))
                model.add(Conv2D(4*f, (3,3), activation='relu'))
                model.add(MaxPool2D(pool_size=(2,2)))
                model.add(Dropout(d))
                model.add(Flatten())
                model.add(Dense(n, activation='relu'))
                model.add(Dropout(d))
                model.add(Dense(6, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(x_eval,y_eval,epochs = 15, batch_size = 200)
                print("Dropout:",d, "num_filters: ",f,"num_nuerons:", n)
                evaluate = model.evaluate(x_test,y_test,batch_size = 200)
                acuraccy.append([evaluate[1],d,f,n])
acuraccy = np.array(acuraccy)
np.save("accuracy",acuraccy[:,0])
bins = [.2,.4,.6,.8,1]
sns.distplot(acuraccy[:,0],bins = bins)
plt.title("Distribution of Accuracies")
