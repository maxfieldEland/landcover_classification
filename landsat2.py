# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:26:13 2018

@author: mgreen13
"""

# REQUIRED PACKAGES
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential
from tensorboard.plugins.pr_curve import summary as pr_summary
from keras.callbacks import TensorBoard
from keras import backend as k
        

# READ IN AND RESHAPE DATA 
x_train = np.load('xTrain.csv.npy')
x_test = np.load('xTest.csv.npy')
x_eval = np.load('xEval.csv.npy')

y_train = np.load('yTrain.csv.npy')
y_test = np.load('yTest.csv.npy')
y_eval = np.load('yEval.csv.npy')


# Veiw sample of data from 
classes = []
ex = yTestSlice[1:16]
for x in ex:
    for ind,x2 in enumerate(x):
        if x2 ==1:
            classes.append(ind)

labels = ["barren","trees", "grassland", "roads", "buildings"," water bodies"]
# PLOT SAMPLE OF LABELED DATA IN SUBPLOTS
fig = plt.figure(figsize = (15,9))
for ind,i in enumerate(range(1,16)):
    ax = fig.add_subplot(3,5,i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(xTestSlice[i,:,:,0:3])
    ax.set_xlabel(labels[classes[i-1]])
plt.draw()

# CREATE TENSORFLOW TENSORBOARD GRAPH
tf.reset_default_graph()

# CREATE MODEL VIA KERAS SEQUENTIAL
model = Sequential()
# ADD LAYERS TO MODEL
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,4)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# INCREASE DROPOUT WITH EACH LAYER
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(4,4)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# DEFINE CALLBACK FOR TENSORBOARD
tbcallback = TensorBoard(log_dir='./Graph/', histogram_freq=1, write_graph=True, write_grads=True)
# FITMODEL, SAVE CALLBACK TO GRAPHS FOLDER
model.fit(x_train, y_train, batch_size=200, epochs=20, verbose=1, validation_data=(x_eval, y_eval), callbacks=[tbcallback])