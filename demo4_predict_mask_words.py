
"""
mask predict
"""

import numpy as np
from tools import tokenizer
import models


text = '中华人民共和国的[MASK][MASK][MASK][MASK][MASK]是习近平'
T = tokenizer.Tokenizer()
words_list, words_array, segment_array, mask_array = T.tokenize(text)

# 2、input
token_input = words_array
token_input = np.asarray([token_input])
segment_input = segment_array
segment_input = np.asarray([segment_input])
mask_input = mask_array
mask_input = np.asarray([mask_input])

# 3、model
model = models.BERT()()


# 4、predict
predicts = model.predict([token_input, segment_input, mask_input])[0]
predicts = predicts.argmax(axis=-1).tolist()
token_dict_inv = T.token_dict_inv
print(list(map(lambda x: token_dict_inv[x], predicts[0][9: 14])))