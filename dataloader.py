from keras.preprocessing.text import Tokenizer
import numpy as np

class TokenList:
    def __init__(self, token_list):
        self.id2token = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + token_list
        self.token2id = {v:k for k, v in enumerate(self.id2token)}

    def id(self, x):
        return self.token2id.get(x, 1)

    def token(self, x):
        return self.id2token[x]

    def length(self):
        return len(self.id2token)

    def start_id(self):
        return 2

    def end_id(self):
        return 3

def pad_to_longest(data, token, max_len=500):
    longest = min(len(max(data, key=len)) + 2, max_len)
    np_array = np.zeros([len(data), longest], dtype='int32')
    np_array[:, 0] = token.start_id()
    for i, x in enumerate(data):
        x  = x[:longest-2]
        for j, z in enumerate(x):
            np_array[i, 1+j] = token.id(z)
        np_array[i, 1+len(x)] = token.end_id()
    return np_array

def make_data(X, y, input_tokens, output_tokens, max_len=200):
    X, y = pad_to_longest(X, input_tokens, max_len), pad_to_longest(y, output_tokens, max_len)
    return X, y

def generate_fake_data(end, num_batches):
    data = np.random.randint(1, end, size=(num_batches, 10))
    return data, data, list(range(1, end))