"""

新闻NER

"""
from tensorflow import keras
import models
from tensorflow.keras import backend as K


with open('data/data_ner/train.txt', 'r') as f:
    text = f.read()
    f.close()

label_dict = {'O': -1, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, "B-ORG": 5, "I-ORG": 6}


max_len = 128

items = text.split('\n\n')

token_inputs = []
segment_inputs = []
y = []

from tools import tokenizer
import numpy as np

T = tokenizer.Tokenizer()
dic = T.token_dict
for item in items:
    item = item.split('\n')
    item = list(map(lambda x: x.split(' '), item))
    sentence = ''.join(list(map(lambda x: x[0], item)))
    token_words, token_array, segment_array, mask_array = T.tokenize(sentence, max_len=max_len)
    token_inputs.append(token_array)
    segment_inputs.append(segment_array)
    label_array = [0] + list(map(lambda x : label_dict[x[1]], item)) + [0]
    y.append(label_array)

token_inputs = np.asarray(token_inputs)
segment_inputs = np.asarray(segment_inputs)
y = keras.preprocessing.sequence.pad_sequences(y, maxlen=max_len, padding='post', truncating='post')
y = K.one_hot(y, len(label_dict))


model = models.BERT(max_len=max_len, base=True)()


in1, in2 = keras.layers.Input((None,)), keras.layers.Input((None,))
x = keras.models.Model(model.inputs[:2], model.outputs)([in1, in2])
x = keras.layers.Dense(len(label_dict))(x)
cc = models.CRF(len(label_dict))
x = cc(x)
model = keras.models.Model([in1, in2], x)

# train
if 0:
    for layer in model.layers:
        layer.trainable = True

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='model_train/nert_ep{epoch:03d}_loss{loss:.3f}.h5',
                                                 monitor='loss',
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1)

    model.compile(optimizer=keras.optimizers.Adam(3e-5), loss=cc.loss, metrics=[cc.viterbi_accuracy])
    model.fit([token_inputs, segment_inputs], y, epochs=10, batch_size=4, validation_split=0.1, callbacks=[checkpoint])

    model.save('model_train/ner_final.h5')

# test
if 1:
    model.load_weights('model_train/nert_ep002_loss0.348.h5')

    sentence = '张三李四和王五都在广东中国人寿上班班'
    token_words, token_array, segment_array, mask_array = T.tokenize(sentence, max_len=max_len)
    token_array = np.asarray([token_array])
    segment_array = np.asarray([segment_array])
    p = model.predict([token_array, segment_array])[0]
    pp = np.argmax(p, -1)
    print(pp)

    def decode(sentence, array, label_dict, type='bert'):
        if type == 'bert':
            array = array[1:len(sentence)]
        label_dict_inv = {label_dict[key]: key for key in label_dict}




