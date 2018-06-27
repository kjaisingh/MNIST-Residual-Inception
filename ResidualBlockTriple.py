#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:38:42 2018

@author: KaranJaisingh
"""

import keras
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Nadam
from keras.layers import BatchNormalization, Convolution2D, Input, merge
from keras.layers.core import Activation, Layer

'''
Keras Customizable Residual Unit
This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit.
'''

def conv_block(feat_maps_out, prev):
    prev = Convolution2D(feat_maps_out, 3, 3, border_mode='same')(prev) 
    prev = Activation('relu')(prev)
    prev = Convolution2D(feat_maps_out, 3, 3, border_mode='same')(prev) 
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Convolution2D(feat_maps_out, 1, 1, border_mode='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([skip, conv], mode='sum') # the residual connection




if __name__ == "__main__":
    
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
    
    
    
    
    cnv1 = Convolution2D(64, 7, 7, subsample=[2,2], activation='relu', border_mode='same')(input_img)
    r1 = Residual(64, 128, cnv1)
    # An example residual unit coming after a convolutional layer. NOTE: the above residual takes the 64 output channels
    # from the Convolutional2D layer as the first argument to the Residual function
    
    r2 = Residual(128, 128, r1)
    r3 = Residual(128, 256, r2)
    
    activationOutput = Activation('relu')(r3)
    
    # Optional MaxPooling layer: 
    # activationOutput = MaxPooling2D((3,3), strides=(1,1), padding='same')(activationOutput)
    
    from keras.layers import Flatten, Dense
    output = Flatten()(activationOutput)
    befOut = Dense(20, activation='relu')(output)
    out = Dense(10, activation='softmax')(befOut) 
    
    
    
    
    model = Model(inputs = input_img, outputs = out)

    from keras.utils.vis_utils import plot_model
    print (model.summary())
    plot_model(model, to_file='model_plot_triple.png', show_shapes=True, show_layer_names=True)

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
    plt.title('Batch vs. Loss (Residual Block Large Network)')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('ResidualBlockTripleLoss.png')
    
    
    
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
    plt.title('Batch vs. Accuracy (Residual Block Large Network)')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('ResidualBlockTripleAcc.png')