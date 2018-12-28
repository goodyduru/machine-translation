import tensorflow as tf

keras = tf.keras
class Attention(keras.layers.Layer):
    """Attention Layer"""

    def __init__(self, hidden_size:int, num_heads:int, dropout:float, mask:bool):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by num of heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask = mask

    def build(self, hidden_size:int):
        self.q_layer = self.add_weight(name="q_layer", shape=(hidden_size, hidden_size), initializer='glorot_uniform', trainable=True)
        self.k_layer = self.add_weight(name="k_layer", shape=(hidden_size, hidden_size), initializer='glorot_uniform', trainable=True)
        self.v_layer = self.add_weight(name="v_layer", shape=(hidden_size, hidden_size), initializer='glorot_uniform', trainable=True)
        self.output_layer = self.add_weight(name="output_layer", shape=(hidden_size, hidden_size), initializer='glorot_uniform', trainable=True)

    def split_heads(self, x):
        """ Split x into different 
        Args:
            x: input with shape [batch_size, length, hidden_size]
        Returns:
            output: tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        depth = self.hidden_size / self.num_heads
        x = tf.reshape(x, [batch_size, length, self.num_heads, depth])
        output = tf.transpose(x, [0, 2, 1, 3])
        return output

    def combine_heads(self, x):
        """Combine x that has been split
         Args:
            x: input with shape [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
            output: tensor with shape [batch_size, length, hidden_size]
        """
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        output = tf.reshape(x, [batch_size, length, self.hidden_size])
        return output

    def call(self, inputs, bias, **kwargs):
        """Apply attention to content of inputs"""
        if not ( isinstance(inputs, list) and len(inputs) == 2):
            raise ValueError('You can call this layer with a list of two tensors, for (key/values and queries)')
        
        K = keras.backend
        x = inputs[0] #[batch_size, length_x, hidden_size]
        y = inputs[1] #[batch_size, length_y, hidden_size]

        q = K.dot(x, self.q_layer)
        k = K.dot(y, self.k_layer)
        v = K.dot(y, self.v_layer)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = self.hidden_size / self.num_heads
        q *= depth ** -0.5

        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = keras.layers.Activation('softmax')(logits)
        weights = keras.layers.Dropout(self.dropout)(weights)
        attention_output = tf.matmul(weights, v)

        #Recombine head
        attention_output = self.combine_heads(attention_output)
        output = K.dot(attention_output, self.output_layer)
        return output


class SelfAttention(Attention):
    def call(self, x, bias):
        return super(SelfAttention, self).call(x, x, bias)