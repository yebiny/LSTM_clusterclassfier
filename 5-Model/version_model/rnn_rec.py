from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout, Masking
from tensorflow.keras.layers import LSTM, BatchNormalization, Activation, TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

def rec_acc(y_true, y_pred):
    # y_true: (batch_size, maxlen, 1)
    # y_pred: (batch_size, maxlen, 1)

    # (batch_size, maxlen, 1)
    is_particle_wise_equal = K.equal(y_true, K.round(y_pred))

    # (batch_size, maxlen)
    is_particle_wise_equal = K.squeeze(is_particle_wise_equal, axis=-1)

    # (batch_size, )
    is_jet_wise_correct = K.all(is_particle_wise_equal, axis=1)

    return K.mean(is_jet_wise_correct)

def rec_crossentropy(y_true, y_pred, from_logits=False):

    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)

    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))

    return K.mean(output, axis=-1)

# basic LSTM
def version_1(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(1, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

# Deeper
def version_2(x_shape):

    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

# Add masking
def version_3(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

# Add LSTM
def version_4(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

# Change Dropout to BatchNor-> Fail. too low acc

# Add dense layer
def version_5(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

# Resize batch
def version_6(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

def version_7(x_shape):

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    
    model.add(TimeDistributed(Dense(32, activation=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('relu'))
   
    model.add(TimeDistributed(Dense(1, activation=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99)) 
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[rec_acc] )
    return model

ZOO = {
    'version_1': version_1,
    'version_2': version_2,
    'version_3': version_3,
    'version_4': version_4,
    'version_5': version_5,
    'version_6': version_6,
    'version_7': version_7,
}

def get_model_fn(model_ver):
    return ZOO[model_ver]


_CUSTOM_OBJECTS = [
    rec_acc,
]
for each in _CUSTOM_OBJECTS:
    key = each.func_name
    get_custom_objects()[key] = each
