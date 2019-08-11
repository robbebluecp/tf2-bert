
import numpy as np
from util_tools import load_model, tokenizer

# 1、先分个词，拿个词向量
text = '中华人民共和国'
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
model = load_model.get_models(base=False)
load_model.get_weights(model, base=False)

# 4、预测
predicts = model.predict([token_input, segment_input, mask_input])[0]

vector = np.mean(predicts, -2)
print(vector.shape)
