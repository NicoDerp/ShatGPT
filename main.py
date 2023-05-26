
import numpy as np
from nltk.tokenize import RegexpTokenizer
from aiLib import *


SENTENCE_DEPTH = 5

with open("data.txt", "r") as f:
    text = f.read().lower()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# A lot fewer words
words = words[:256]

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
unique_word_index_reverse = dict((i, c) for i, c in enumerate(unique_words))
# print(words)
# print(unique_words)

next_word = []
prev_words = []
for j in range(len(words) - SENTENCE_DEPTH):
    prev_words.append(words[j:j + SENTENCE_DEPTH])
    next_word.append(words[j + SENTENCE_DEPTH])
# print(prev_words)
# print(next_word)

# print(unique_word_index)

# Very inefficient one-shot
X = np.zeros((len(prev_words), SENTENCE_DEPTH, len(unique_words)), dtype=int)
Y = np.zeros((len(next_word), len(unique_words)), dtype=int)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_word[i]]] = 1

ai = AI(layers=[
            InputLayer((len(unique_words),)),
            # LSTMLayer(len(unique_words)),
            LSTMLayer(50),
            # FFLayer(50, activation="Sigmoid"),
            # FFLayer(50, activation="ReLU"),
            FFLayer(len(unique_words), activation="Softmax")
        ],
        loss="CategoricalCrossEntropy",
        # loss="MSE",
        # optimizer="RMSprop",
        # optimizer="Momentum",
        optimizer="Adam",
        learningRate=0.001)

# ai = AI.load("shatgpt.model")
# ai.learningRate = 0.001
# ai.train(X, Y, epochs=200, mbSize=64, shuffle=True)
ai.train(X, Y, epochs=500, mbSize=X.shape[0], shuffle=True)

ai.save("shatgpt.model")


encoded = []
testWords = words[0:5]
print("Testing", testWords)

for w in testWords:
    enc = np.zeros(len(unique_words))
    enc[unique_word_index[w]] = 1
    encoded.append(enc)

predicted = ai.predictNextWord(encoded, unique_word_index_reverse, n=2)

print(f"AI predicted word '{predicted}'")

