import collections
import numpy as np
from keras.utils import to_categorical
import string
import re
import unicodedata
from keras.preprocessing.text import Tokenizer

def unicode_to_ascii(w):
    return ''.join([c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn'])

#clean and normalize string
def clean_lines(lines, to_ascii=False):
    cleaned = list()
    for line in lines:
        if to_ascii:
            line = unicode_to_ascii(line.strip())
        line = re.sub(r"([?.!,¿])", r" \1 ", line)
        line = re.sub(r'[" "]+', " ", line)
        line = re.sub(r"[^a-zA-Z1-9?.!,¿]+", " ", line)
        line = line.rstrip().strip()
        line = "\t " + line + " \n"
        cleaned.append(line)
    return cleaned


def get_all_characters(text_list):
    characters = []
    for text in text_list:
        for char in text:
            if char not in characters:
                characters.append(char)
    return sorted(characters)


def get_vocab(characters):
    vocab = collections.defaultdict(lambda: 0, {v:k for k,v in enumerate(characters)})

    return vocab

def get_inv_vocab(vocab):
    inv_vocab = {}
    for k, v in vocab.items():
        inv_vocab[v] = k

    return inv_vocab

def string_to_int(string, vocab, length):
    if len(string) > length:
        string = string[:length]
    int_value = list(map(lambda x: vocab.get(x, '<unk>'), string))
    if len(string) < length:
        int_value += [vocab['<pad>']] * (length - len(string))
    return int_value

def preprocess_data(data, vocab, length):
    data = np.array([string_to_int(i, vocab, length) for i in data])
    one_hot_encode = np.array(list(map(lambda i: to_categorical(i, num_classes=len(vocab)), data)))
    return data, one_hot_encode

def preprocess_input_data(texts, vocab, length, num_tokens):
    one_hot_encode = np.zeros([len(texts), length, num_tokens])
    for i, text in enumerate(texts):
        for t, char in enumerate(text):
            one_hot_encode[i, t, vocab[char]] = 1
    return one_hot_encode

def preprocess_target_data(texts, vocab, length, num_tokens):
    one_hot_encode = np.zeros([len(texts), length, num_tokens])
    for i, text in enumerate(texts):
        for t, char in enumerate(text):
            if t > 0:
                one_hot_encode[i, t - 1, vocab[char]] = 1
    return one_hot_encode

def decode_sequence(input_seq, encoder_model, num_decoder_tokens, igbo_vocab, decoder_model, inv_igbo_vocab, Ty):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, igbo_vocab['\t']] = 1

    stop = False
    decoded_sentence = ''
    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_output_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = inv_igbo_vocab[sampled_output_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > Ty:
            stop = True
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_output_index] = 1

        states_value = [h, c]
    return decoded_sentence

def get_short_sentences(dataset, Tx, Ty):
    english_text = dataset['en'].tolist()
    igbo_text = dataset['ig'].tolist()
    short_english_text = []
    short_igbo_text = []
    for english, igbo in zip(english_text, igbo_text):
        if len(english) <= Tx and len(igbo) <= Ty:
            short_english_text.append(english)
            short_igbo_text.append(igbo)
    return short_english_text, short_igbo_text