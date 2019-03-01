import keras.backend as K 
from keras.engine import InputSpec
from keras.layers import Layer, Embedding
from keras.callbacks import Callback

class LayerNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

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
        mean = K.mean(x, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=self.axis, keepdims=True)
        norm = (x - mean) / (variance + self.epsilon)
        return norm * self.scale + self.bias

class Embeddings(Embedding):
    def call(self, inputs):
        out = super(Embeddings, self).call(inputs)
        out *= (self.output_dim ** 0.5)
        return out

class LRSchedulerPerStep(Callback):
    def __init__(self, model_size, warmup=4000):
        self.basic = model_size**-0.5
        self.warm = warmup**-1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
        K.set_value(self.model.optimizer.lr, lr)

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