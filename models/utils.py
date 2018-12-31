import collections
import numpy as np
from keras.utils import to_categorical

def get_all_characters(text_list):
    characters = []
    for text in text_list:
        for char in text:
            if char not in characters:
                characters.append(char)
    return sorted(characters)


def get_vocab(characters):
    characters.insert(0, '<bos>')
    characters.insert(1, '<pad>')
    characters.insert(2, '<eos>')
    characters.insert(3, '<unk>')
    vocab = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(characters)})

    return vocab

def string_to_int(string, vocab, length):
    if len(string) > length:
        string = string[:length]
    int_value = list(map(lambda x: vocab.get(x, '<unk>'), string))
    if len(string) < length:
        int_value += [vocab['<pad>']] * (length - len(string))
    return int_value

def preprocess_data(data, vocab, length):
    data = np.array([string_to_int(i, vocab, length) for i in data])
    print(data)
    one_hot_encode = np.array(list(map(lambda i: to_categorical(i, num_classes=len(vocab)), data)))
    return data, one_hot_encode