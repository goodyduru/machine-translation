import keras.backend as K 
from keras.layers import Layer
from keras.engine import InputSpec
import numpy as np

def get_attention_decoder_mask(length):
    """Calculate bias for decoder that maintains model's autoregressive property.
    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.
    Args:
        length: int length of sequences in batch.
    Returns:
        float tensor of shape [1, 1, length, length]
    """
    _NEG_INF = -1e9
    valid_locs = K.tf.matrix_band_part(K.ones((length, length)), -1, 0)
    valid_locs = K.reshape(valid_locs, (1, 1, length, length))
    decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias

class Masking(Layer):
    def __init__(self, padding: int=0, decoder_mask: bool = False, **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.padding = padding
        self.decoder_mask = decoder_mask
        self._NEG_INF = -1e9
        
    def build(self, input_shape):
        self.batch_size, self.length = input_shape
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.length})
        self.built = True

    def call(self, x):
        if self.decoder_mask:
            length = K.shape(x)[1]
            self.mask = get_attention_decoder_mask(length)
            return self.mask
        else:
            padding = K.cast(K.equal(x, self.padding), dtype=K.floatx())
            self.mask = padding * self._NEG_INF
            self.mask = K.expand_dims(K.expand_dims(self.mask, axis=1), axis=1)
            return self.mask

    def compute_output_shape(self, input_shape):
        if self.decoder_mask:
            return (1, 1, self.length, self.length)
        else:
            return(self.batch_size, 1, 1, self.length)

    def get_config(self):
        config = {
            'padding': self.padding,
            'decoder_mask': self.decoder_mask
        }
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    i = Input(shape=(104, ), dtype='float32')
    dec = Masking(decoder_mask=True)(i)
    model = Model(inputs=i, outputs=dec)
    print(model.summary())