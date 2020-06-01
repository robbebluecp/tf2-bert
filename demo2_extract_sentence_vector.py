
"""

extract sentence vector

"""

import numpy as np
from tools import tokenizer
import models
from tensorflow import keras


text = '中华人民共和国'
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
model = models.BERT(base=False)()
in1_ = keras.layers.Input((None, ))
in2_ = keras.layers.Input((None, ))
new_model = keras.models.Model(model.inputs[:2], model.get_layer('Extract').output)


# 4、predict
predicts = new_model.predict([token_input, segment_input, mask_input])

print(predicts.shape, predicts[0])



