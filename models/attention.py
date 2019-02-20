from keras.layers import Layer, Dropout
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec
import keras.backend as K
import tensorflow as tf

class Attention(Layer):
    """Attention Layer"""

    def __init__(self, output_dim: int, num_heads: int, 
                dropout: float=0.,
                activation=None,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs
                ):
        if output_dim % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by num of heads")
        super(Attention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self.batch_size, self.timesteps, self.input_dim = input_shape
        depth = (self.output_dim//self.num_heads)
        self.W_q = self.add_weight(shape=(self.input_dim, self.output_dim),
                                    name='W_q',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        
        self.b_q = self.add_weight(shape=(self.output_dim,),
                                    name='b_q',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.W_k = self.add_weight(shape=(self.input_dim, self.output_dim),
                                    name='W_k',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.b_k = self.add_weight(shape=(self.output_dim,),
                                    name='b_k',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.W_v = self.add_weight(shape=(self.input_dim, self.output_dim),
                                    name='W_v',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.b_v = self.add_weight(shape=(self.output_dim,),
                                    name='b_v',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                    name='W_o',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.b_o = self.add_weight(shape=(self.output_dim,),
                                    name='b_o',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.bias = self.add_weight(shape=(self.timesteps,),
                                    name='bias',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.input_spec = InputSpec(min_ndim=3, axes={-1: self.input_dim})
        self.built = True

    def split_heads(self, x):
        """ Split x into different 
        Args:
            x: input with shape [batch_size, timesteps, output_dim]
        Returns:
            output: tensor with shape [batch_size, num_heads, timesteps, output_dim/num_heads]
        """
        depth = self.output_dim // self.num_heads
        x = K.reshape(x, (-1, self.timesteps, self.num_heads, depth))
        output = K.permute_dimensions(x, [0, 2, 1, 3])
        return output

    def combine_heads(self, x):
        """Combine x that has been split
         Args:
            x: input with shape [batch_size, num_heads, timesteps, output_dim/num_heads]
        Returns:
            output: tensor with shape [batch_size, timesteps, output_dim]
        """
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        output = K.reshape(x, (-1, self.timesteps, self.output_dim))
        return output

    def call(self, x, training=None):
        q = K.dot(x, self.W_q) + self.b_q
        k = K.dot(x, self.W_k) + self.b_q
        v = K.dot(x, self.W_v) + self.b_v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = self.output_dim // self.num_heads
        q *= depth ** -0.5
        k = K.permute_dimensions(k, (0, 1, 3, 2))
        logits = K.batch_dot(q, k)
        logits += self.bias
        weights = K.softmax(logits)

        if 0. < self.dropout < 1.:
            def dropped_inputs():
                return K.dropout(weights, self.dropout)
            weights = K.in_train_phase(weights, dropped_inputs, training)
        
        attention_output = K.batch_dot(weights, v)

        #Recombine head
        attention_output = self.combine_heads(attention_output)
        output = K.dot(attention_output, self.W_o) + self.b_o
        return output

    def compute_output_shape(self, input_shape):
        return (None, self.timesteps, self.output_dim)

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'output_dim': self.output_dim
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    from keras.layers import Input, LSTM
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    i = Input(shape=(100,104), dtype='float32')
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
    dec = Attention(32, 4, 0.1)(enc)
    model = Model(inputs=i, outputs=dec)
    print(model.summary())