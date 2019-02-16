from keras import regularizers, activations, constraints, initializers
from keras.layers.recurrent import Recurrent
from keras.layers import TimeDistributed, LSTM, Bidirectional, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.engine import InputSpec
import keras.backend as K

def time_distributed_dense(input_tensor, weight, bias=None, timesteps=None, input_dim=None, output_dim=None, dropout=None, training=None):
    """Apply t.weight + bias for every t of timesteps of input
        input_tensor: input tensor shape = (batch num, timestep, input_dim)
        weight: weight tensor = (input_dim, output_dim)
        bias: optional bias
        dropout: dropout value
        training: training phase boolean
    """
    if timesteps is None:
        timesteps = K.shape(input_tensor)[1]
    if input_dim is None:
        input_dim = K.shape(input_tensor)[2]
    if output_dim is None:
        output_dim = K.shape(weight)[1]

    if dropout is not None and 0. < dropout < 1.:
        #apply dropout at every timestep
        ones = K.ones_like(K.reshape(input_tensor[:, 0, :], (-1, input_dim)))
        dropout_tensor = K.dropout(ones, dropout)
        dropout_tensor_with_timestep = K.repeat(dropout_tensor, timesteps)
        input_tensor = K.in_train_phase(input_tensor * dropout_tensor_with_timestep, input_tensor, training=training)
    
    #collapse timestep and batch num together
    input_tensor = K.reshape(input_tensor, (-1, input_dim))
    input_tensor = K.dot(input_tensor, weight)
    if bias is not None:
        input_tensor = K.bias_add(input_tensor, bias)
    
    output_tensor = K.reshape(input_tensor, (-1, timesteps, output_dim))
    return output_tensor


class AttentionLayer(Recurrent):
    def __init__(self, units, output_dim,
        activation='tanh',
        return_attention_weights=False,
        name='AttentionLayer',
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs):
        self.units = units
        self.output_dim = output_dim
        self.return_attention_weights = return_attention_weights
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionLayer, self).__init__(kwargs)
        self.name = name
        self.return_sequences = True

    def build(self, input_shape):
        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionLayer, self).reset_states()
        
        self.states = [None, None] #output, state

        """
        Matrices for memory gate
        """
        self.W_c = self.add_weight(shape=(self.output_dim, self.units),
                                    name='W_c',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.U_c = self.add_weight(shape=(self.units, self.units),
                                    name='U_c',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        
        self.C_c = self.add_weight(shape=(self.input_dim, self.units),
                                    name='C_c',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.b_c = self.add_weight(shape=(self.units, ),
                                    name='b_c',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        """
        Matrices for update gate z
        """
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                    name='W_z',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.U_z = self.add_weight(shape=(self.units, self.units),
                                    name='U_z',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                    name='C_z',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.b_z = self.add_weight(shape=(self.units, ),
                                    name='b_z',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        """
        Matrices for reset gate r
        """
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                    name='W_r',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.U_r = self.add_weight(shape=(self.units, self.units),
                                    name='U_r',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                    name='C_r',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.b_r = self.add_weight(shape=(self.units, ),
                                    name='b_r',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        """
        Matrices for output gate o
        """
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                    name='W_o',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                    name='U_o',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                    name='C_o',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)

        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                    name='b_o',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        """
        Matrices for aligning
        """
        self.V_a = self.add_weight(shape=(self.units,),
                                    name='V_a',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.W_a = self.add_weight(shape=(self.units, self.units),
                                    name='W_a',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                    name='U_a',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.b_a = self.add_weight(shape=(self.units, ),
                                    name='b_a',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        #Weight for creating initial state
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                    name='W_s',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        
        self.input_spec = [InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        self.x_seq = x
        self.uh = time_distributed_dense(self.x_seq, self.U_a, self.b_a, timesteps=self.timesteps, input_dim=self.input_dim, output_dim=self.units)
        return super(AttentionLayer, self).call(x)

    def get_initial_state(self, inputs):
        s0 = activations.tanh(K.dot(inputs[:, 1], self.W_s))
        y0 = K.zeros_like(inputs) # (batch_size, timesteps, input_dim)
        y0 = K.sum(y0, axis=(1, 2)) # (batch_size,)
        y0 = K.expand_dims(y0) # (batch_size, 1)
        y0 = K.tile(y0, [1, self.output_dim]) #(batch_size, output_dim)
        return [y0, s0]

    def step(self, x, states):
        y_prev, s_prev = states
        s_all = K.repeat(s_prev, self.timesteps)
        Wa_s_all = K.dot(s_all, self.W_a)
        et = K.dot(activations.tanh(Wa_s_all + self.uh), K.expand_dims(self.V_a))
        #et_sum = K.sum(K.exp(et), axis=1)
        #et_sum_repeated = K.repeat(et_sum, self.timesteps)
        #a_current = et_sum / et_sum_repeated #shape batch_size, timestep, 1
        a_current = activations.softmax(et)
        context = K.squeeze(K.batch_dot(a_current, self.x_seq, axes=1), axis=1)
        #calculate reset gate
        r_current = activations.sigmoid(K.dot(y_prev, self.W_r)
                                 + K.dot(s_prev, self.U_r) 
                                 + K.dot(context, self.C_r) 
                                 + self.b_r)

        #calculate update gate
        z_current = activations.sigmoid(K.dot(y_prev, self.W_z)
                                + K.dot(s_prev, self.U_z)
                                + K.dot(context, self.C_z)
                                + self.b_z)

        #calculate s tilde
        s_tilde = activations.tanh(
                                K.dot(y_prev, self.W_c) 
                                + K.dot((r_current * s_prev), self.U_c)
                                + K.dot(context, self.C_c) 
                                + self.b_c)

        s_current = (1 - z_current) * s_prev + z_current * s_tilde

        #calculate output
        y_current = activations.sigmoid(
                                        K.dot(y_prev, self.W_o) 
                                        + K.dot(s_current, self.U_o) 
                                        + K.dot(context, self.C_o)
                                        + self.b_o)  
        
        if self.return_attention_weights:
            return a_current, [y_current, s_current]
        else:
            return y_current, [y_current, s_current]
    
    def compute_output_shape(self, input_shape):
        if self.return_attention_weights:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)
    

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_attention_weights': self.return_attention_weights
        }
        
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def custom_model(num_encoder_tokens, num_decoder_tokens, lstm_dim, timesteps):
    input_layer = Input(shape=(timesteps, num_encoder_tokens))
    encoder_lstm = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    enc = encoder_lstm(input_layer)

    attention_decoder = AttentionLayer(lstm_dim, num_decoder_tokens, name='attention')
    outputs = attention_decoder(enc)

    model = Model(inputs=input_layer, outputs=outputs)
    return model