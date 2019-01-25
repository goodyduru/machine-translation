from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM
from keras.layers import Multiply, RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
import numpy as np

def custom_softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    else:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    
def initialize_shared_weights(Tx, n_s, vocab):
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor_1 = Dense(10, activation='tanh')
    densor_2 = Dense(1, activation='relu')
    activator = Activation(custom_softmax, name='attentional_weights')
    dotor = Dot(axes=1)
    post_attention_LSTM = LSTM(n_s, return_state=True)
    output_layer = Dense(len(vocab), activation='softmax')
    return repeator, concatenator, densor_1, densor_2, activator, dotor, post_attention_LSTM, output_layer

def one_step_attention(a, s_prev, repeator, concatenator, densor_1, densor_2, activator, dotor):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    * -- Layers
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor_1(concat)
    energies = densor_2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context

def custom_model(Tx, Ty, n_a, n_s, english_vocab_size, igbo_vocab_size, repeator, concatenator, densor_1, densor_2, activator, dotor, post_attention_LSTM, output_layer):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    english_vocab_size -- size of the python dictionary "english_vocab"
    igbo_vocab_size -- size of the python dictionary "igbo_vocab"

    Returns:
    model -- Keras model instance
    """
    X = Input(shape=(Tx, english_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    for t in range(Ty):
        context = one_step_attention(a, s, repeator, concatenator, densor_1, densor_2, activator, dotor)
        s, _, c = post_attention_LSTM(context, initial_state=[s, c])
        output = output_layer(s)
        outputs.append(output)
    model = Model([X, s0, c0], outputs=outputs)
    return model