
import numpy as np
from nltk.tokenize import RegexpTokenizer
from aiLib import *


SENTENCE_DEPTH = 5

with open("data.txt", "r") as f:
    text = f.read().lower()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# A lot fewer words
words = words[:512]

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
# print(words)
# print(unique_words)

dataset = []

# for i in range(len(words) - SENTENCE_DEPTH):
#     dataset.append()

next_word = []
prev_words = []
for j in range(len(words) - SENTENCE_DEPTH):
    prev_words.append(words[j:j + SENTENCE_DEPTH])
    next_word.append(words[j + SENTENCE_DEPTH])
print(prev_words[1])
print(next_word[1])

# Very inefficient one-shot
X = np.zeros((len(prev_words), SENTENCE_DEPTH, len(unique_words)), dtype=int)
Y = np.zeros((len(next_word), len(unique_words)), dtype=int)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_word[i]]] = 1

dataset = []
for prevs, cur in zip(X, Y):
    sentence = []
    for i in range(len(prevs)-1):
        sentence.append((prevs[i], prevs[i+1]))
    sentence.append((prevs[-1], cur))
    dataset.append(sentence)

ai = AI(layers=[
            InputLayer((len(unique_words),)),
            LSTMLayer(len(unique_words)),
            # FFLayer(6, activation="Sigmoid"),
            FFLayer(len(unique_words), activation="Softmax")
        ],
        # loss="CategoricalCrossEntropy",
        loss="MSE",
        optimizer="RMSprop",
        learningRate=0.01)

# dataset = [
#     [
#         [np.array([0, 0, 1]), np.array([0, 1])],
#         #[np.array([4, 1, 2]), np.array([1, 0])]
#     ],
#     # [
#     #     [np.array([5, 2, 1]), np.array([1, 0.0])],
#     #     [np.array([0, 1, 2]), np.array([1, 0.0])]
#     # ],
# ]

ai.train(dataset, epochs=10, mbSize=128, shuffle=True)

ai.save("shatgpt.model")


