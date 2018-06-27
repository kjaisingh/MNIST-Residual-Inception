#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:37:41 2018

@author: KaranJaisingh
"""

import keras
from keras.layers import Input
import matplotlib.pyplot as plt




from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

from keras.utils import np_utils
y_train =  np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

input_img = Input(shape = (28, 28, 1))




from keras.layers import Conv2D, MaxPooling2D
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding='same',  activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)

output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1_2 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
tower_1_2 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1_2)
tower_2_2 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
tower_2_2 = Conv2D(64, (5,5), padding='same',  activation='relu')(tower_2_2)
tower_3_2 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output)
tower_3_2 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3_2)
tower_4_2 = Conv2D(64, (1,1), padding='same', activation='relu')(output)

outputN = keras.layers.concatenate([tower_1_2, tower_2_2, tower_3_2, tower_4_2], axis = 3)




from keras.layers import Flatten, Dense
outputN = Flatten()(outputN)
befOut = Dense(20, activation='relu')(outputN)
out    = Dense(10, activation='softmax')(befOut)




from keras.models import Model
model = Model(inputs = input_img, outputs = out)

from keras.utils.vis_utils import plot_model
print (model.summary())
plot_model(model, to_file='model_plot_double.png', show_shapes=True, show_layer_names=True)




class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
lossHistory = LossHistory()

class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('acc'))
accHistory = AccHistory()




from keras.optimizers import SGD
epochs = 1
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[lossHistory, accHistory])

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




historyLossPlot = []
historyLossBatch = []
historyLossArray = lossHistory.losses
i = 0
for j in historyLossArray:
    if(i%10 == 0):
            historyLossPlot.append(j)
            historyLossBatch.append(i)
    i += 1
plt.figure()        
plt.plot(historyLossBatch, historyLossPlot)
plt.grid(True)
plt.axhline(y = 0, color = 'k')
plt.axvline(x = 0, color = 'k')
plt.title('Batch vs. Loss (Inception Module Large Network)')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()
plt.savefig('DIAGRAMInceptionModuleDoubleLoss.png')



historyAccPlot = []
historyAccBatch = []
historyAccArray = accHistory.losses
i = 0
for j in historyAccArray:
    if(i%10 == 0):
            historyAccPlot.append(j)
            historyAccBatch.append(i)
    i += 1
plt.figure()
plt.plot(historyAccBatch, historyAccPlot)
plt.grid(True)
plt.axhline(y = 0, color = 'k')
plt.axvline(x = 0, color = 'k')
plt.title('Batch vs. Accuracy (Inception Module Large Network)')
plt.xlabel('Batch Number')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('DIAGRAMInceptionModuleDoubleAcc.png')
