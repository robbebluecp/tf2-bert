

import numpy as np
from util_tools import load_model, tokenizer

# 1、先分个词，拿个词向量
text1 = '今天晚饭吃的比较多'
text2 = '散步的时候肚子不太舒服'
# text2 = '欧拉公式是世上最美的公式'
T = tokenizer.Tokenizer()
words_list, words_array, segment_array, mask_array = T.tokenize(text1, text2)


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
predicts = model.predict([token_input, segment_input, mask_input])[1]
predicts = predicts.argmax(axis=-1).tolist()[0]
print('text2有木有可能是text1的下一句:', not bool(predicts))
