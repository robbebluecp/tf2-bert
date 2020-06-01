import numpy as np
import config

f = open(config.bert_path)

dic = {}

index = 0
for i in f.readlines():
    if i not in dic:
        dic[i.strip()] = index
        index += 1


max_len = 128
with open('data/data_ner/train.txt', 'r') as f:
    text = f.read()
    f.close()

label_dict = {'O': -1, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, "B-ORG": 5, "I-ORG": 6}
items = text.split('\n\n')


x_input = []
y_input = []
for item in items:
    item = item.split('\n')
    x_tmp = []
    y_tmp = []
    for sub_item in item:
        sub_item = sub_item.split()
        x_tmp.append(dic.get(sub_item[0], 100))
        y_tmp.append(label_dict[sub_item[1]])
    x_input.append(x_tmp)
    y_input.append(y_tmp)



from tensorflow import keras
import tensorflow.keras.backend as K
import models

x_input = keras.preprocessing.sequence.pad_sequences(x_input, maxlen=max_len, padding='post', truncating='post')
y_input_ = keras.preprocessing.sequence.pad_sequences(y_input, maxlen=max_len, padding='post', truncating='post')
y_input = K.one_hot(y_input_, len(label_dict))


model = keras.Sequential()
model.add(keras.layers.Embedding(len(dic), 256, mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(keras.layers.Dense(len(label_dict)))
cc = models.CRF(len(label_dict))
model.add(cc)

checkpoint = keras.callbacks.ModelCheckpoint(filepath='model_train/ner_ep{epoch:03d}-loss{loss:.3f}.h5',
                                             monitor='loss',
                                             save_weights_only=False,
                                             save_best_only=True,
                                             period=2)


model.compile(optimizer=keras.optimizers.Adam(), loss=cc.loss, metrics=[cc.viterbi_accuracy])
model.fit(x_input, y_input, epochs=10, callbacks=[checkpoint], batch_size=64)
model.save_weights('model_train/ner_final.h5')


model.load_weights('model_train/ner_ep006-loss1.744.h5')
p = model.predict(x_input[-1:])
print(np.argmax(p, -1))
print(y_input_[-1])

