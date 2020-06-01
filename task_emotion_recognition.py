"""

情感分析，文本分类，语义相似度计算均可使用如下方式进行训练


"""

import models
import numpy as np
from tensorflow import keras
from tools import tokenizer


T = tokenizer.Tokenizer()

with open('data/date_comment/train.txt', 'r', encoding='utf8') as f:
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
    _, token_array, segment_array, mask_array = T.tokenize(text)
    token_input.append(token_array)
    segment_input.append(segment_array)
    mask_input.append(mask_array)


token_input = np.asarray(token_input)
segment_input = np.asarray(segment_input)
mask_input = np.asarray(mask_input)
labels = np.asarray(labels, dtype=np.int)



in1_ = keras.layers.Input((None, ))
in2_ = keras.layers.Input((None, ))

base_model = models.BERT()()


x = keras.models.Model(base_model.inputs[:2], base_model.get_layer('Extract').output)([in1_, in2_])
x = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model([in1_, in2_], x)
for layer in model.layers:
    layer.trainable = True


tensorboard = keras.callbacks.TensorBoard()
checkpoint = keras.callbacks.ModelCheckpoint(filepath='model_train/ep{epoch:03d}-loss{loss:.3f}.h5',
                                             monitor='val_loss',
                                             save_weights_only=False,
                                             save_best_only=True,
                                             period=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1)

model.compile(optimizer=keras.optimizers.Adam(5e-5), loss=keras.losses.binary_crossentropy, metrics=['loss', 'acc'])


model.fit(x=[token_input, segment_input],
          y=labels, validation_split=0.1, epochs=5, batch_size=16,
          callbacks=[checkpoint, early_stopping])

model.save_weights('demo_comment.weights')

# text = '标准间太差 房间还不如3星的 而且设施非常陈旧.建议酒店把老的标准间从新改善.'
# _, token_array, segment_array, mask_array = T.tokenize(text, max_len=128)


# token_input = np.asarray([token_array])
#
# segment_input = np.asarray([segment_array])
# mask_input = np.asarray([mask_array])

# a = model.predict([token_input, segment_input, mask_input])
# print(a)
