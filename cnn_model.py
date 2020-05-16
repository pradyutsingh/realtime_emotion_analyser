import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import os.path 

from tensorflow.keras import layers
from keras.models import load_model


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def define_model():
    model = tf.keras.Sequential()

#1
    model.add(layers.Conv2D(32, 3, input_shape=(48, 48, 1), padding='same', 
                 activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, 3, padding='same', 
                 activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))

#2
    model.add(layers.Conv2D(64, 3, padding='same', 
                 activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 3, padding='same', 
                 activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(layers.Dropout(0.25))

# 3
    model.add(layers.Conv2D(128, 3, padding='same', 
                 activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, 3, padding='same', 
                 activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(layers.Dropout(0.25))

# FC layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(7))
    model.add(layers.Activation('softmax'))
    
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def model_weights(model):
    # load already saved model if needed
    if os.path.exists('C:/Users/KIIT/Desktop/emotion_detect/weights1.h5'):
        model.load_weights('C:/Users/KIIT/Desktop/emotion_detect/weights1.h5')
    else:
        print('No model to load !')
    return model
    
