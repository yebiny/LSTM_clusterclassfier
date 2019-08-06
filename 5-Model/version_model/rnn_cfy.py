from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout, Masking
from tensorflow.keras.layers import LSTM, BatchNormalization, Activation

def version_1(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def version_2(x_shape):

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


def version_3(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(x_shape[1],x_shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


def version_4(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


def version_5(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def version_6(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    
    model.add(Dense(64, activation=None))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('relu'))
    
    model.add(Dense(32, activation=None))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('relu'))
    
    model.add(Dense(1, activation=None))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('sigmoid'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def version_7(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(512, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def version_8(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    
    model.add(Dense(64, activation=None))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('relu'))
    
    model.add(Dense(32, activation=None))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('relu'))
    
    model.add(Dense(1, activation=None))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('sigmoid'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

ZOO = {
    'version_1': version_1,
    'version_2': version_2,
    'version_3': version_3,
    'version_4': version_4,
    'version_5': version_5,
    'version_6': version_6,
    'version_7': version_7,
    'version_8': version_8,
}

def get_model_fn(model_ver):
    return ZOO[model_ver]
