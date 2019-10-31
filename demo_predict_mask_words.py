
"""

完形填空，这个水准应该比高考生水准要高

"""

import keras
import numpy as np
from keras_bert.util_tools import load_model, tokenizer

# 1、先分个词，拿个词向量
text = '中华人民共和国的[MASK][MASK]是习近平'
T = tokenizer.Tokenizer()
words_list, words_array, segment_array, mask_array = T.tokenize(text)


# 2、构造输入
token_input = words_array
token_input = np.asarray([token_input])
segment_input = segment_array
segment_input = np.asarray([segment_input])
mask_input = mask_array
mask_input = np.asarray([mask_input])

# 3、加载模型
model = load_model.get_models()
load_model.get_weights(model)

# 4、预测
predicts = model.predict([token_input, segment_input, mask_input])[0]

predicts = predicts.argmax(axis=-1).tolist()
token_dict_inv = T.token_dict_inv
print(list(map(lambda x: token_dict_inv[x], predicts[0][9: 11])))