"""

情感分析，文本分类，语义相似度计算均可使用如下方式进行训练


"""



import os
from keras_bert.util_tools import *
import numpy as np
import keras




base_dir = os.getcwd()
T = tokenizer.Tokenizer(type='cls')
d = download.Downloader()
d.distribute_task('hotel_comments')

with open(base_dir + '/data/hotel_comments/hotel_comments.txt', 'r', encoding='gbk') as f:
    items = f.readlines()
    labels = []
    sentences = []
    for item in items:
        item = item.split(',', 1)
        labels.append(int(item[0]))
        sentences.append(item[1].strip())

    f.close()

token_input = []
segment_input = []
mask_input = []
for text in sentences:
    if len(text) > 512:
        text = text[:510]
    _, token_array, segment_array, mask_array = T.tokenize(text, max_len=100)
    token_input.append(token_array)
    segment_input.append(segment_array)
    mask_input.append(mask_array)


token_input = np.asarray(token_input)
segment_input = np.asarray(segment_input)
mask_input = np.asarray(mask_input)
labels = np.asarray(labels, dtype=np.int)

#
#
#
sample_num = list(range(2000)) + list(range(len(token_input)))[-2000:]
sample_num = np.asarray(sample_num, int)


in1_ = keras.layers.Input((None, ))
in2_ = keras.layers.Input((None, ))

base_model = load_model.get_models(max_len=100)
load_model.get_weights(base_model, max_len=100)


x = keras.models.Model(base_model.inputs[:2], base_model.get_layer('Extract').output)([in1_, in2_])
x = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model([in1_, in2_], x)
for layer in model.layers:
    layer.trainable = True


model.compile(optimizer=keras.optimizers.Adam(5e-5), loss=keras.losses.binary_crossentropy, metrics=['acc'])
model.fit(x=[token_input[sample_num], segment_input[sample_num]],
          y=labels[sample_num], validation_split=0.1, epochs=5, batch_size=8)

# text = '标准间太差 房间还不如3星的 而且设施非常陈旧.建议酒店把老的标准间从新改善.'
# _, token_array, segment_array, mask_array = T.tokenize(text, max_len=128)


# token_input = np.asarray([token_array])
#
# segment_input = np.asarray([segment_array])
# mask_input = np.asarray([mask_array])

# a = model.predict([token_input, segment_input, mask_input])
# print(a)
