import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed

if tf.test.is_gpu_available(cuda_only=True):
    from tensorflow.keras.layers import CuDNNLSTM as LSTM
else:
    from tensorflow.keras.layers import LSTM


def build_model(x_shape):
    model = Sequential()
    model.add(TimeDistributed(Dense(32), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(1,  return_sequences=True)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

#labels=ydataset
#
#logits = tf.layers.dense(inputs=inputs, units=num_output_labels, activation=None)
#inputs= rnnoutput
#num_output_labels=
#
#losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    #logits=tf.layers.dens(input=(25,5))
    #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    #labels=10,
    #logits=10,
    #)

    #model.compile( loss='mse',optimizer='sgd')
    #model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


