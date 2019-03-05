from keras.layers import Dropout, Add, Input, Dense, Embedding
from keras.layers import Activation, Lambda, TimeDistributed
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
import keras.backend as K
import numpy as np
from .attention import MultiHeadAttention, MultiHeadSelfAttention
from .extras import LayerNormalization, Embeddings
from .ffn import FeedForwardNetwork, Padding
from .masking import Masking
from .position import PositionEncoding

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
        self.attention_norm = LayerNormalization()
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
        ffn_out = self.ffn_layer(ffn_out)
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

class Transformer():
    def __init__(self, source_tokens, target_tokens, model_size, ffn_size, num_layers, num_heads, dropout=0.1):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.model_size = model_size
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.decoder_model = None

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

        self.target_layer = TimeDistributed(Dense(target_tokens.length()))

    def compile(self, optimizer='adam'):
        source_input = Input(shape=(None, ), dtype='int32')
        target_input = Input(shape=(None, ), dtype='int32')

        target_true = Lambda(lambda x: x[:, 1:])(target_input)
        target_seq = Lambda(lambda x: x[:, :-1])(target_input)

        encoder_output = self.encoder(source_input)
        decoder_output = self.decoder(target_seq, source_input, encoder_output)
        output = self.target_layer(decoder_output)

        def get_loss(args):
            y_pred, y_true = args
            y_true = K.cast(y_true, 'int32')
            loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
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

        loss = Lambda(get_loss, name='loss')([output, target_true])
        ppl = Lambda(K.exp)(loss)
        accu = Lambda(get_accu)([output, target_true])

        self.model = Model(inputs=[source_input, target_input], outputs=loss)
        self.model.add_loss([loss])

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('perplexity')
        self.model.metrics_tensors.append(ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(accu)

    def make_source_sequence_matrix(self, input_sequence):
        source_sequence = np.zeros([1, len(input_sequence) + 3], dtype='int32')
        source_sequence[0, 0] = self.source_tokens.start_id()
        input_sequence = text_to_word_sequence(input_sequence)
        for i, z in enumerate(input_sequence):
            source_sequence[0, i+1] = self.source_tokens.id(z)
        source_sequence[0, len(input_sequence) + 1] = self.source_tokens.end_id()
        return source_sequence

    def make_decode_model(self):
        source_input = Input(shape=(None, ), dtype='int32')
        target_input = Input(shape=(None, ), dtype='int32')

        encoder_output = self.encoder(source_input)
        self.encoder_model = Model(inputs=source_input, outputs=encoder_output)

        encoder_input = Input(shape=(None, self.model_size))
        decoder_output = self.decoder(target_input, source_input, encoder_input)
        final_output = self.target_layer(decoder_output)
        self.decoder_model = Model(inputs=[source_input, target_input, encoder_input], outputs=final_output)

        self.encoder_model.compile('adam', 'mse')
        self.decoder_model.compile('adam', 'mse')

    def decode_sequence(self, input_sequence, len_limit=50):
        if self.decoder_model is None:
            self.make_decode_model()
        source_sequence = self.make_source_sequence_matrix(input_sequence)
        encoded_seq = self.encoder_model.predict_on_batch(source_sequence)

        decoded_tokens = []
        target_sequence = np.zeros([1, self.target_tokens.length()], 'int32')
        target_sequence[0, 0] = self.target_tokens.start_id()

        for i in range(len_limit - 1):
            output = self.decoder_model.predict_on_batch([source_sequence, target_sequence, encoded_seq])
            index = np.argmax(output[0, i, :], -1)
            token = self.target_tokens.token(index)
            decoded_tokens.append(token)
            if index == self.target_tokens.end_id():
                break
            target_sequence[0, i+1] = index
        return " ".join(decoded_tokens[:-1]) 