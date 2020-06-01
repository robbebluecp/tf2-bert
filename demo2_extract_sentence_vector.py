
"""

抽取句向量，这里我用的均值抽取

"""

import numpy as np
from tools import tokenizer
import models
from tensorflow import keras


# 1、先分个词，拿个词向量
text = '中华人民共和国'
T = tokenizer.Tokenizer(type='cls')
words_list, words_array, segment_array, mask_array = T.tokenize(text)


# 2、构造输入
token_input = words_array
token_input = np.asarray([token_input])
segment_input = segment_array
segment_input = np.asarray([segment_input])
mask_input = mask_array
mask_input = np.asarray([mask_input])

# 3、加载模型
model = models.BERT(base=False)()
in1_ = keras.layers.Input((None, ))
in2_ = keras.layers.Input((None, ))

new_model = keras.models.Model(model.inputs[:2], model.get_layer('Extract').output)


# 4、预测
predicts = new_model.predict([token_input, segment_input, mask_input])

print(predicts.shape, predicts[0])



