import keras.backend as K 
from keras.layers import Layer, Embedding


class LayerNormalization(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.scale = self.add_weight(shape=(self.hidden_size,),
                                name='scale',
                                initializer='ones',
                                trainable=True)
        
        self.bias = self.add_weight(shape=(self.hidden_size,),
                                name='bias',
                                initializer='zeros',
                                trainable=True)
        self.built = True

    def call(self, x):
        epsilon = K.constant(1e-6, dtype=K.floatx())
        mean = K.mean(x, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=self.axis, keepdims=True)
        norm = (x - mean) / K.sqrt(variance + epsilon)
        return norm * self.scale + self.bias

class Embeddings(Embedding):
    def call(self, inputs):
        out = super(Embeddings, self).call(inputs)
        out *= (self.output_dim ** 0.5)
        return out

if __name__ == "__main__":
    from keras.layers import Input, LSTM, Embedding
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    inp = Input(shape=(100,), dtype='float32')
    emb = Embeddings(1000, 64)(inp)
    encode = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(emb)
    dec = LayerNormalization()(encode)
    model = Model(inputs=inp, outputs=dec)
    print(model.summary())