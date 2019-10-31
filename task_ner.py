"""

新闻NER

"""
import keras
import keras_contrib


keras.utils.get_custom_objects().update(
    {'crf_loss': keras_contrib.losses.crf_losses.crf_loss,
     'crf_viterbi_accuracy': keras_contrib.metrics.crf_viterbi_accuracy})


# DIY NETWORK
with open('data/news_ner/train.txt', 'r') as f:
    text = f.read()
    f.close()

label_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC':4, "B-ORG":5, "I-ORG":6}

"""

# DIT network #

items = text.split('\n\n')

token_inputs = []
segment_inputs = []
y = []


for item in items:
    item = item.split('\n')
    item = list(map(lambda x: x.split(' '), item))
    try:
        sentence = ''.join(list(map(lambda x: x[0], item)))
        label_array = list(map(lambda x: label_dict[x[1]], item))
        token_array = [dic[i] for i in sentence]
        token_inputs.append(token_array)
        y.append(label_array)
    except:
        continue

token_inputs = keras.preprocessing.sequence.pad_sequences(token_inputs, maxlen=64, value=0)
y = keras.preprocessing.sequence.pad_sequences(y, maxlen=64, value=-1)
y = np.expand_dims(y, -1)


model = keras.Sequential()
model.add(keras.layers.Embedding(len(dic), 256, mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(keras_contrib.layers.CRF(len(label_dict), sparse_target=True))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(), loss=keras_contrib.losses.crf_loss, metrics=[keras_contrib.metrics.crf_viterbi_accuracy])
model.fit(token_inputs, y, epochs=3, batch_size=256, validation_split=0.1)
model.save('tmp.h5')

"""

items = text.split('\n\n')

token_inputs = []
segment_inputs = []
y = []

from keras_bert.util_tools import *
import numpy as np

T = tokenizer.Tokenizer()
dic = T.token_dict

for item in items:
    item = item.split('\n')
    item = list(map(lambda x: x.split(' '), item))
    try:
        sentence = ''.join(list(map(lambda x: x[0], item)))
        _, token_array, segment_array, _ = T.tokenize(sentence, max_len=64)
        token_inputs.append(token_array)
        segment_inputs.append(segment_array)
        # cls index = 0, sep index = -1
        label_array = [0] + list(map(lambda x: label_dict[x[1]], item)) + [0]
        y.append(label_array)
    except:
        continue

token_inputs = np.asarray(token_inputs)
segment_inputs = np.asarray(segment_inputs)
y = keras.preprocessing.sequence.pad_sequences(y, maxlen=64)
y = np.expand_dims(y, axis=-1)

model = load_model.get_models(base=True, max_len=64)
load_model.get_weights(model, max_len=64, base=True)

in1, in2 = keras.layers.Input((None,)), keras.layers.Input((None,))
x = keras.models.Model(model.inputs[:2], model.outputs)([in1, in2])
x = keras_contrib.layers.CRF(len(label_dict), sparse_target=True)(x)
model = keras.models.Model([in1, in2], y)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=keras.optimizers.Adam(), loss=keras_contrib.losses.crf_loss, metrics=[keras_contrib.metrics.crf_viterbi_accuracy])
model.fit([token_inputs, segment_inputs], y, epochs=1, batch_size=8, validation_split=0.1)