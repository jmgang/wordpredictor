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
import keras.backend as K

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5

path = 'ALLnew.fl'
text = open(path, encoding="utf8").read().lower().rstrip()
text = re.sub(r'([^\s\w]|_)+', '', text)
text = text.replace("\n", '')
#print('corpus length:', len(text))

# Create a Character-to-Index Mapping
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#print(f'unique chars: {len(chars)}')

# Set the Sequence Length and Window Size (Step)
SEQUENCE_LENGTH = 40
step = 3

# Get the Feature (Sentences) and the target labels (Next Characters)
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
#print(f'num training examples: {len(sentences)}')

# Convert the training data into One-Hot-Encoding vectors
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# # #BUILD THE MODEL
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))
#
# # Train the model
# optimizer = Adam(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history
#
# # #Save the model
# model.save('keras_model_100.h5')
# pickle.dump(history, open("history_100.p", "wb"))


# Encode the input texts
def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
    return x

# prepare_input("This is an example of input for our LSTM".lower())

# Encode the sample into one hot encoding
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

# Get the predicted next character
def predict_completion(text):
    original_text = text
    generated = text
    completion = ''

    #Load the Model
    model = load_model('keras_model.h5')


    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion.replace(u'\xa0', u' ')

# Complete the predicted word
def predict_completions(text, n=3):
    ##Load the Model
    model = load_model('keras_model.h5')
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


# quotes = [
#       "Ang Transformation ay isang iskultura ng tatlong magkakapatong",
#       "Isang magandang tanawin na may natural na lawa at halos 2.5 kilometro",
#       "Mula rito isa itong perpektong lugar na para pagmasdan",
#       "Ginamit ito upang ibahay ang isang magandang koleksyon ng pilak at jeweled",
#       "Sa mga orihinal na manggagawa, ang mga Igorot at Hapones ang hinangaan para sa"
#  ]
#
# for q in quotes:
#       q = re.sub(r'([^\s\w]|_)+', '', q)
#       seq = q[:40].lower()
#       print(seq)
#       print(predict_completions(seq, 5))
#       print()

# evaluation()
