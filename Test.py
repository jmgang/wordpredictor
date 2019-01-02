import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
import re


def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.

    return x

prepare_input("This is an example of input for our LSTM".lower())

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion.replace(u'\xa0', u' ')

def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

quotes = [
    "Ang Transformation ay isang iskultura ng tatlong magkakapatong",
    "Isang  magandang tanawin na may natural na lawa at halos 2.5 kilometro",
    "Mula rito isa itong perpektong lugar na para pagmasdan",
    "Ginamit ito upang ibahay ang isang magandang koleksyon ng pilak at jeweled",
    "Sa mga orihinal na manggagawa, ang mga Igorot at Hapones ang hinangaan para sa"
]

for q in quotes:
    q = re.sub(r'([^\s\w]|_)+', '', q)
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()
