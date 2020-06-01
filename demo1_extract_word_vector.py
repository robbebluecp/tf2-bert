"""

extract word vector

"""


import numpy as np
from tools import tokenizer
import models


text = '你好'
T = tokenizer.Tokenizer()
words_list, words_array, segment_array, mask_array = T.tokenize(text)

# inputs
token_input = words_array
token_input = np.asarray([token_input])
segment_input = segment_array
segment_input = np.asarray([segment_input])
mask_input = mask_array
mask_input = np.asarray([mask_input])

# model
model = models.BERT(base=True)()

# predict
predicts = model.predict([token_input, segment_input, mask_input])
predicts = predicts[0]

for i, token in enumerate(words_list[:len(text) + 2]):
    print(token, predicts[i].shape)