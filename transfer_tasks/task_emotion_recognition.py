import os
from util_tools import tokenizer
from util_tools import load_model
import keras
import numpy as np

base_dir = os.path.dirname(__file__).rsplit('/', 1)[0]
T = tokenizer.Tokenizer()

with open(base_dir + '/data/hotel_comments/hotel_comments.txt', 'r', encoding='gbk') as f:
    items = f.readlines()
    labels = []
    sentences = []
    for item in items:
        item = item.split(',')
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
labels = np.expand_dims(labels, -1)
print(token_input.shape)

model = load_model.get_models()
load_model.get_weights(model)
last_later = model.get_layer('Cls-Dense').output
for layer in model.layers:
    layer.trainable = False

outputs = keras.layers.Dense(2)(last_later)
inputs = model.inputs
model = keras.models.Model(inputs, outputs)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(x=[token_input, segment_input, mask_input], y=labels, validation_split=0.1, epochs=10, batch_size=32)
model.save('tmp.h5')