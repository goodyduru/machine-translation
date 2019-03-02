import math
import keras.backend as K
from keras.layers import Layer
from keras.engine import InputSpec

def get_position_encoding(timesteps: int, hidden_size: int, min_timescale: float=1.0, max_timescale: float=1.0e4):
    """Returns positional encoding
    Calculates the positional encoding as a mixture of sin and cosine with 
    increasing wavelengths.

    Args:
        timesteps: Sequence length
        hidden_size: hidden size 
        min_timescale: Minimum scale to apply to each position
        max_timescale: Maximum scale to apply to each position

    Returns:
        Tensor with shape [timesteps, hidden_size]
    """
    position = K.arange(0, timesteps, dtype=K.floatx())
    num_timescales = K.cast(hidden_size // 2, K.floatx())
    log_timescale_increment = K.log(max_timescale / min_timescale / (num_timescales - 1))
    inv_timescales = min_timescale * K.exp(K.arange(0, num_timescales, dtype=K.floatx()) * -log_timescale_increment)
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)

    return K.expand_dims(signal, 0)

class PositionEncoding(Layer):
    def __init__(self, min_timescale: float=1.0, max_timescale: float=10e4,  **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        
    def build(self, input_shape):
        _, _, hidden_size = input_shape
        
        self.input_spec = InputSpec(min_ndim=3, axes={-1: hidden_size})
        self.built = True

    def call(self, x):
        timesteps = K.shape(x)[1]
        hidden_size = K.shape(x)[2]
        self.signal = get_position_encoding(timesteps, hidden_size, self.min_timescale, self.max_timescale)
        return x + self.signal

    def compute_output_shape(self, input_shape):
        _, timesteps, hidden_size = input_shape
        return (None, timesteps, hidden_size)

    def get_config(self):
        config = {
            'min_timescale': self.min_timescale,
            'max_timescale': self.max_timescale
        }
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    from keras.layers import Input, LSTM
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    i = Input(shape=(100,104), dtype='float32')
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
    dec = PositionEncoding()(enc)
    model = Model(inputs=i, outputs=dec)
    print(model.summary())