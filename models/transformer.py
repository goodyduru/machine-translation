from keras.layers import Dropout, Add, Input, Dense
from keras.layers import Activation, Lambda, TimeDistributed
from keras.models import Model
import keras.backend as K
from attention import MultiHeadAttention, MultiHeadSelfAttention
from extras import LayerNormalization, Embeddings
from ffn import FeedForwardNetwork, Padding
from masking import Masking
from position import PositionEncoding

class EncoderLayer:
    """ A single encoder layer made up of a self attention layer and a feed 
        forward network layer.
    """
    def __init__(self, index, num_heads, model_size, ffn_size, dropout):
        """ Initialize the encoder layer
        Args:
            index: index of the encoder layer, important for naming
            num_heads: number of heads in the multi headed self attention layer
            model_size: size of the weights in the attention layer
            ffn_size: size of the weights in the feed forward network layer
            dropout: dropout value
        """
        self.num_heads = num_heads
        self.model_size = model_size
        self.ffn_size = ffn_size
        self.dropout = dropout
        attention_name = 'encoder_attention_' + str(index + 1)
        self.attention_norm = LayerNormalization()
        self.ffn_norm = LayerNormalization()
        self.attention_layer = MultiHeadSelfAttention(output_dim=model_size, num_heads=num_heads, dropout=dropout, name=attention_name)
        ffn_name = 'encoder_ffn_' + str(index + 1)
        self.ffn_layer = FeedForwardNetwork(ffn_size, relu_dropout=dropout, allow_pad=True, name=ffn_name)

    def __call__(self, x, mask, pad):
        """Returns the output of an encoder layer
        Args:
            x: input of the encoder
            mask: mask for the encoder input
            pad: pad for the encoder input

        Returns:
            The output of the encoder layer
        """
        #Start Attention Network
        attention_out = self.attention_norm(x)
        attention_out = self.attention_layer([attention_out, mask])
        attention_out = Dropout(self.dropout)(attention_out)
        ffn_in = Add()([x, attention_out])

        #Start Feed Forward Network
        ffn_out = self.ffn_norm(ffn_in)
        ffn_out = self.ffn_layer([ffn_out, pad])
        ffn_out = Dropout(self.dropout)(ffn_out)
        ffn_out = Add()([ffn_in, ffn_out])
        return ffn_out

class EncoderStack():
    """A stack of encoder layers """
    def __init__(self, emb, mask, pad, pos, num_layers=6, num_heads=8, model_size=256, ffn_size=256, dropout=0.1):
        """Initializes the encoder stack of num_layers encoder layers
        Args:
            emb: embedding layer for the encoder input
            mask: mask layer for the encoder input
            pad: padding layer for the encoder input
            pos: positional encoder layer for the encoder input
            num_layers: number of encoder layers
            num_head: number of head for the attention layers in each of the  encoder layer
            model_size: size of the attention layers
            ffn_size: size of the feed forward network layers
            dropout: dropout value
        """
        self.emb = emb
        self.mask = mask
        self.pad = pad
        self.pos = pos
        self.dropout = dropout
        self.encoders = []
        self.output_normalization = LayerNormalization(name='encoder')
        for i in range(num_layers):
            encoder = EncoderLayer(i, num_heads, model_size, ffn_size, dropout)
            self.encoders.append(encoder)

    def __call__(self, x):
        """ Passes the encoder input through all the encoder layers
        Args:
            x: encoder input

        Returns:
            out: encoder output
        """
        source_pad = self.pad(x)
        source_mask = self.mask(x)
        x = self.emb(x)
        x = self.pos(x)
        x = Dropout(self.dropout)(x)
        for encoder in self.encoders:
            x = encoder(x, source_mask, source_pad)
        out = self.output_normalization(x)
        return out
    

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

    out = LayerNormalization(name='encoder')(x)
    return out

class DecoderLayer():
    """ A single decoder layer made up of a self attention layer, an encoder 
        decoder attention layer a feed forward network layer
    """
    def __init__(self, index, num_heads, model_size, ffn_size, dropout):
        """ Initialize the encoder layer
        Args:
            index: index of the encoder layer, important for naming
            num_heads: number of heads in the multi headed self attention layer
            model_size: size of the weights in the attention layer
            ffn_size: size of the weights in the feed forward network layer
            dropout: dropout value
        """
        self.num_heads = num_heads
        self.model_size = model_size
        self.ffn_size = ffn_size
        self.dropout = dropout

        # Names
        self_attention_name = 'decoder_attention_' + str(index + 1)
        attention_name = 'enc_dec_attention_' + str(index + 1)

        #Normalization
        self.self_attention_norm = LayerNormalization()
        self.attention_norm = LayerNormalization
        self.ffn_norm = LayerNormalization()

        #Layers
        self.self_attention_layer = MultiHeadSelfAttention(output_dim=model_size, num_heads=num_heads, dropout=dropout, name=self_attention_name)
        self.attention_layer = MultiHeadAttention(output_dim=model_size, num_heads=num_heads, dropout=dropout, name=attention_name)
        ffn_name = 'decoder_ffn_' + str(index + 1)
        self.ffn_layer = FeedForwardNetwork(ffn_size, relu_dropout=dropout, allow_pad=False, name=ffn_name)

    def __call__(self, x, encoder_output, decoder_mask, encoder_mask):
        """Returns the output of the decoder layer
        Args:
            x: input of the decoder
            encoder_output: Output of the encoder stack
            decoder_mask: mask for the decoder input
            encoder_mask: mask for the encoder input

        Returns:
            The output of the decoder layer
        """
        #Start Self Attention Network
        self_attention_out = self.self_attention_norm(x)
        self_attention_out = self.self_attention_layer([self_attention_out, decoder_mask])
        self_attention_out = Dropout(self.dropout)(self_attention_out)
        attention_in = Add()([x, self_attention_out])

        #Encoder Decoder Attention Network
        attention_out = self.attention_norm(attention_in)
        attention_out = self.attention_layer([attention_out, encoder_output, encoder_mask])
        attention_out = Dropout(self.dropout)(attention_out)
        ffn_in = Add()([attention_in, attention_out])

        #Start Feed Forward Network
        ffn_out = self.ffn_norm(ffn_in)
        ffn_out = self.ffn_layer([ffn_out])
        ffn_out = Dropout(self.dropout)(ffn_out)
        ffn_out = Add()([ffn_in, ffn_out])
        return ffn_out

class DecoderStack():
    """A stack of decoder layers"""
    def __init__(self, emb, encoder_mask, decoder_mask, pos, num_layers=6, num_heads=8, model_size=256, ffn_size=256, dropout=0.1):
        """Initializes the decoder stack of num_layers decoder layers
        Args:
            emb: embedding layer for the decoder input
            encoder_mask: mask layer for the encoder input
            decoder_mask: mask layer for the decoder input
            pos: positional encoder layer for the decoder input
            num_layers: number of encoder layers
            num_head: number of head for the attention layers in each of the decoder layer
            model_size: size of the attention layers
            ffn_size: size of the feed forward network layers
            dropout: dropout value
        """
        self.emb = emb
        self.encoder_mask = encoder_mask
        self.decoder_mask = decoder_mask
        self.pos = pos
        self.dropout = dropout
        self.decoders = []
        self.output_normalization = LayerNormalization(name='decoder_norm')
        #self.lambda_layer = Lambda(lambda x: K.temporal_padding(x[:, :-1, :], (1, 0)))
        for i in range(num_layers):
            decoder = DecoderLayer(i, num_heads, model_size, ffn_size, dropout)
            self.decoders.append(decoder)

    def __call__(self, target, source, encoder_output):
        """ Passes the decoder input through all the decoder layers
        Args:
            target: decoder input
            source: encoder input
            encoder_output = output of encoder layer

        Returns:
            out: decoder output
        """
        target_mask = self.decoder_mask(target)
        source_mask = self.encoder_mask(source)
        target_embedding = self.emb(target)
        #target_embedding = self.lambda_layer(target_embedding)
        decoder_inputs = self.pos(target_embedding)
        x = Dropout(self.dropout)(decoder_inputs)
        for decoder in self.decoders:
            x = decoder(x, encoder_output, target_mask, source_mask)
        out = self.output_normalization(x)
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

class Transformer():
    def __init__(self, source_tokens, target_tokens, source_len, target_len, model_size, ffn_size, num_layers, num_heads, dropout=0.1):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.source_len = source_len
        self.target_len = target_len
        self.model_size = model_size
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = self.dropout

        #source layers
        source_emb = Embeddings(source_tokens.length(), model_size)
        source_pad = Padding()
        source_mask = Masking()
        source_pos = PositionEncoding()

        #target layers
        target_emb = Embeddings(target_tokens.length(), model_size)
        target_mask = Masking(decoder_mask=True)
        target_pos = PositionEncoding()

        #encoder
        self.encoder = EncoderStack(source_emb, source_mask, source_pad, source_pos, num_layers, num_heads, model_size, ffn_size, dropout)

        #decoder
        self.decoder = DecoderStack(target_emb, source_mask, target_mask, target_pos, num_layers, num_heads, model_size, ffn_size, dropout)

        self.target_layer = TimeDistributed(Dense(target_tokens.length, activation='softmax'))

    def compile(self, optimizer='adam'):
        source_input = Input(shape=(self.source_len, ), dtype='int32')
        target_input = Input(shape=(self.target_len, ), dtype='int32')

        target_true = Lambda(lambda x: x[:, 1:])(target_input)
        target_seq = Lambda(lambda x: x[:, :-1])(target_input)

        encoder_output = self.encoder(source_input)
        decoder_output = self.decoder(target_seq, source_input, encoder_output)
        output = self.target_layer(decoder_output)

        def get_loss(args):
            y_pred, y_true = args
            y_true = K.cast(y_true, 'int32')
            loss = K.sparse_categorical_crossentropy(y_true, y_pred)
            mask = K.cast(K.not_equal(y_true, 0), K.floatx())
            loss = K.sum(loss * mask, -1) / K.sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(args):
            y_pred, y_true = args
            y_true = K.cast(y_true, 'int32')
            mask = K.cast(K.not_equal(y_true, 0), K.floatx())
            y_pred = K.cast(K.argmax(y_pred, -1), 'int32')
            corr = K.cast(K.equal(y_true, y_pred), K.floatx())
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss)([output, target_true])
        ppl = Lambda(K.exp)(loss)
        accu = Lambda(get_accu)([output, target_true])

        self.model = Model(inputs=[source_input, target_input], outputs=loss)
        self.model.add_loss([loss])
        self.output_model = Model(inputs=[source_input, target_input], outputs=output)

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('perplexity')
        self.model.metrics_tensors.append(ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(accu)



        


    
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
    target_embedding = Lambda(lambda x: K.temporal_padding(x[:, :-1, :], (1, 0)))(target_embedding)
    decoder_inputs = PositionEncoding()(target_embedding)
    decoder_inputs = Dropout(dropout)(decoder_inputs)
    decoder_output = decoder(decoder_inputs, encoder_output, source_mask, target_mask, num_layers, num_heads, model_size, dropout)

    output = TimeDistributed(Dense(target_vocab, name='decoder', activation='softmax'))(decoder_output)

    model = Model(inputs=[source, target], outputs=output)
    return model

if __name__ == "__main__":
    model = transformer(15, 20, 10000, 12000, 256, 3, 8, 0.1)
    print(model.summary())