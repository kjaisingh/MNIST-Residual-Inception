#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:56:53 2018

"""
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import Convolution2D, Input, merge
from keras.layers.core import Activation, Layer

# Convolutional layer part of Residual Block
def convolutional_block(num_output_feats, bef):
    bef = Convolution2D(num_output_feats, 3, 3, border_mode='same')(bef) 
    bef = Activation('relu')(bef)
    bef = Convolution2D(num_output_feats, 3, 3, border_mode='same')(bef) 
    return bef

# The skip function used for a Residual connection
def skip_block(num_input_feats, num_output_feats, bef):
    if num_input_feats != num_output_feats:
        bef = Convolution2D(num_output_feats, 1, 1, border_mode='same')(bef)
    return bef 


# Residual Block
def Residual(num_input_feats, num_output_feats, bef_layer):
    skip = skip_block(num_input_feats, num_output_feats, bef_layer)
    conv = convolutional_block(num_output_feats, bef_layer)

    print('Residual block mapping '+str(num_input_feats)+' channels to '+str(num_output_feats)+' channels built')
    return merge([skip, conv], mode='sum') # the residual connection





if __name__ == "__main__":
    
    # Importing and preprocessing the dataset
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
    
    
    
    # Instantiating a Residual unit
    cnv1 = Convolution2D(64, 7, 7, subsample=[2,2], activation='relu', border_mode='same')(input_img)
    r1 = Residual(64, 128, cnv1)
    activationOutput = Activation('relu')(r1)

    # Feeding the output into Fully Connected layers    
    from keras.layers import Flatten, Dense
    output = Flatten()(activationOutput)
    befOut = Dense(20, activation='relu')(output)
    out = Dense(10, activation='softmax')(befOut) 
    
    # Constructing the model
    model = Model(inputs = input_img, outputs = out)
    
    
    
    # Visualizing the model built
    from keras.utils.vis_utils import plot_model
    print (model.summary())
    plot_model(model, to_file='model_plot_single.png', show_shapes=True, show_layer_names=True)




    # Classes that track the history of the model's training
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
    
    
    
    # Training the model
    from keras.optimizers import SGD
    epochs = 1
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[lossHistory, accHistory])
    
    model.save('my_model.h5')



    # Testing the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    
    # Visualizing the accuracy and loss plots
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
    plt.title('Batch vs. Loss (Residual Block Small Network)')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('ResidualBlockSingleLoss.png')
    
    
    
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
    plt.title('Batch vs. Accuracy (Residual Block Small Network)')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('ResidualBlockSingleAcc.png')
