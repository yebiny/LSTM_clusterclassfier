import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout, Masking
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

def build_model(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


