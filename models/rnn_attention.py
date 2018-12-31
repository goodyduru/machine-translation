from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM
from keras.layers import Multiply, RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
import numpy as np