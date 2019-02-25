from keras.layers import Dropout, Add, Input
from keras.models import Model
from attention import MultiHeadAttention, MultiHeadSelfAttention
from extras import LayerNormalization, Embeddings, PadDecoder
from ffn import FeedForwardNetwork, Padding
from masking import Masking
from position import PositionEncoding

def attention_sub_layer(x, mask, name, num_heads=8, size=256, dropout=0.1, memory=None):
    out = LayerNormalization()(x)
    if memory is None:
        out = MultiHeadSelfAttention(output_dim=size, num_heads=num_heads, dropout=dropout, name=name)([out, mask])
    else:
        out = MultiHeadAttention(output_dim=size, num_heads=num_heads, dropout=dropout, name=name)([out, memory, mask])
    out = Dropout(dropout)(out)
    return Add()([x, out])

def feedforward_sub_layer(x, name, pad=None, size=256, dropout=0.1):
    out = LayerNormalization()(x)
    if pad is not None:
        out = FeedForwardNetwork(size, relu_dropout=dropout, allow_pad=True, name=name)([x, pad])
    else:
        out = FeedForwardNetwork(size, relu_dropout=dropout, allow_pad=False, name=name)(x)
    out = Dropout(dropout)(out)
    return Add()([x, out])

def encoder(x, pad, mask, num_layers=6, num_heads=8, size=256, dropout=0.1):
    for i in range(num_layers):
        attention_name = 'encoder_attention_' + str(i + 1)
        ffn_name = 'encoder_ffn_' + str(i + 1)
        x = attention_sub_layer(x, mask, attention_name, num_heads, size, dropout)
        x = feedforward_sub_layer(x, ffn_name, pad, size, dropout)

    out = LayerNormalization()(x)
    return out

def decoder(x, encoder_outputs, mask, decoder_mask, num_layers=6, num_heads=8, size=256, dropout=0.1):
    for i in range(num_layers):
        attention_name = 'decoder_attention_' + str(i + 1)
        enc_dec_name = 'encoder_decoder_attention_' + str(i + 1)
        ffn_name = 'decoder_ffn_' + str(i + 1)
        x = attention_sub_layer(x, decoder_mask, attention_name, size=size, dropout=dropout)
        x = attention_sub_layer(x, mask, enc_dec_name, size=size, dropout=dropout, memory=encoder_outputs)
        x = feedforward_sub_layer(x, ffn_name, size=size, dropout=dropout)
    out = LayerNormalization()(x)
    return out
    
def transformer(source_length, target_length, source_vocab, target_vocab, model_size, num_layers, num_heads, dropout):
    source = Input(shape=(source_length,))
    source_padding = Padding()(source)
    source_mask = Masking()(source)
    source_embedding = Embeddings(source_vocab, model_size)(source)
    encoder_inputs = PositionEncoding()(source_embedding)
    encoder_inputs = Dropout(dropout)(encoder_inputs)
    encoder_output = encoder(encoder_inputs, source_padding, source_mask, num_layers, num_heads, model_size, dropout)

    target = Input(shape=(target_length,))
    target_mask = Masking(decoder_mask=True)(target)
    target_embedding = Embeddings(target_vocab, model_size)(target)
    target_embedding = PadDecoder()(target_embedding)
    decoder_inputs = PositionEncoding()(target_embedding)
    decoder_inputs = Dropout(dropout)(decoder_inputs)
    decoder_output = decoder(decoder_inputs, encoder_output, source_mask, target_mask, num_layers, num_heads, model_size, dropout)

    model = Model(inputs=[source, target], outputs=decoder_output)
    return model

if __name__ == "__main__":
    model = transformer(15, 20, 10000, 12000, 256, 3, 8, 0.1)
    print(model.summary())