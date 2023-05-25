
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


ai = AI.load("shatgpt.model")

encoded = []
testWords = words[1:6]
print("Testing", testWords)

for w in testWords:
    enc = np.zeros(len(unique_words))
    enc[unique_word_index[w]] = 1
    encoded.append(enc)

predicted = ai.predictNextWord(encoded, unique_word_index_reverse, n=10)

print(f"AI predicted word '{predicted}'")

