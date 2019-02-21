import keras.backend as K 
from keras import initializers, regularizers, constraints
from keras.layers import Layer, Dense
from keras.engine import InputSpec
import tensorflow as tf

class Padding(Layer):
    def __init__(self, padding: int=0, **kwargs):
        super(Padding, self).__init__(**kwargs)
        self.padding = padding
        
    def build(self, input_shape):
        self.batch_size, self.length = input_shape
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.length})
        self.built = True

    def call(self, x):
        padding = K.cast(K.equal(x, self.padding), dtype=K.floatx())
        return padding

    def compute_output_shape(self, input_shape):\
        return(self.batch_size, self.length)

    def get_config(self):
        config = super(Padding, self).get_config()
        config['padding'] = self.padding
        return config

class FeedForwardNetwork(Layer):
    def __init__(self, filter_size: int,
                relu_dropout: float = 0.1,
                allow_pad: bool = True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.allow_pad = allow_pad
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        if self.allow_pad and not (isinstance(input_shape, list) and ( len(input_shape)  == 2 )):
            raise ValueError(
                'You must call this layer passing a list of two tensors'
                '(for keys/values and queries) if you want padding')
        if self.allow_pad:
            self.batch_size, self.length, self.hidden_size = input_shape[0]
            self.input_spec = [InputSpec(min_ndim=3, axes={-1: self.hidden_size}), InputSpec(min_ndim=2, axes={-1: self.length})]
        else:
            self.batch_size, self.length, self.hidden_size = input_shape
            self.input_spec = [InputSpec(min_ndim=3, axes={-1: self.hidden_size})]
        
        self.filter_weight = self.add_weight(shape=(self.hidden_size, 
                                self.filter_size),
                                name='filter_weight',
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
        
        self.filter_bias = self.add_weight(shape=(self.filter_size,),
                                name='filter_bias',
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint)

        self.output_weight = self.add_weight(shape=(self.filter_size,
                                self.hidden_size),
                                name='output_weight',
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)

        self.output_bias = self.add_weight(shape=(self.hidden_size,),
                                name='output_bias',
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint)
        self.built = True

    def call(self, input_tensor, training=None):
        if self.allow_pad and not (isinstance(input_tensor, list) and ( len(input_tensor)  == 2 )):
            raise ValueError(
                'You must call this layer passing a list of two tensors'
                '(for keys/values and queries) if you want padding')

        if isinstance(input_tensor, list) and (len(input_tensor) == 2) :
            x, padding = input_tensor
        else:
            x, padding = input_tensor, None
        
        batch_size = K.shape(x)[0]
        padding = None if not self.allow_pad else padding

        if padding is not None:
            # Reshape padding to [batch_size*length]
            pad_mask = K.reshape(padding, (-1,))
            nonpad_ids = K.cast(K.tf.where(pad_mask < 1e-9), dtype='int32')

            # Reshape x to [batch_size*length, hidden_size] to remove padding
            x = K.reshape(x, (-1, self.hidden_size))
            x = K.gather(x, indices=nonpad_ids)
            # Reshape to 3 dimensions
            x = K.squeeze(x, axis=1)
            x.set_shape((None, self.hidden_size))
            x = K.expand_dims(x, axis=0)
        output = K.dot(x, self.filter_weight) + self.filter_bias
        
        if 0. < self.relu_dropout < 1.:
            def dropped_inputs():
                return K.dropout(output, self.relu_dropout)
            output = K.in_train_phase(output, dropped_inputs, training)

        output = K.dot(output, self.output_weight) + self.output_bias
        if padding is not None:
            output = K.squeeze(output, axis=0)
            output = K.tf.scatter_nd(
                indices = nonpad_ids,
                updates = output,
                shape=[batch_size*self.length, self.hidden_size]
            )
            output = K.reshape(output, (batch_size, self.length, self.hidden_size))
        return output

    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.length, self.hidden_size)

    def get_config(self):
        config = {
            'filter_size': self.filter_size,
            'relu_dropout': self.relu_dropout,
            'allow_pad': self.allow_pad
        }
        base_config = super(FeedForwardNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    from keras.layers import Input, LSTM, Embedding
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    inp = Input(shape=(100,), dtype='float32')
    padding = Padding()(inp)
    emb = Embedding(1000, 64)(inp)
    encode = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(emb)
    dec = FeedForwardNetwork(60)([encode, padding])
    model = Model(inputs=inp, outputs=dec)
    print(model.summary())