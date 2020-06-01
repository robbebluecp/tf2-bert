
"""
relation predict
"""

import numpy as np
import models
from tools import tokenizer


text1 = '明天会下雨'
text2 = '我们不去踢足球了'
T = tokenizer.Tokenizer()
words_list, words_array, segment_array, mask_array = T.tokenize(text1, text2)


# 2、inputs
token_input = words_array
token_input = np.asarray([token_input])
segment_input = segment_array
segment_input = np.asarray([segment_input])
mask_input = mask_array
mask_input = np.asarray([mask_input])

# 3、model
model = models.BERT()()

# 4、predict
predicts = model.predict([token_input, segment_input, mask_input])[1]
print(predicts)
predicts = predicts.argmax(axis=-1).tolist()[0]
print('text2有木有可能是text1的下一句:', not predicts)
