from keras_bert.models import *
import tensorflow as tf
import json
import os
import numpy as np


base_dir = os.getcwd()
ckpt_file = base_dir + '/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
config_file = base_dir + '/data/chinese_L-12_H-768_A-12/bert_config.json'


def get_config():
    with open(config_file, 'r') as f:
        config = json.load(f)
        f.close()
    return config


def get_models(base=False, max_len=512):
    config = get_config()
    # embedding vocab size
    vocab_size = config['vocab_size']  # 21128
    # max length each sentence
    max_len = min(config['max_position_embeddings'], max_len)  # 512
    # embedding output_dim
    emb_dim = config['hidden_size']  # 768
    #
    dropout_rate = config['hidden_dropout_prob']  # 0.1
    # attention block num
    block_num = config['num_hidden_layers']  # 12
    # head num
    n_head = config['num_attention_heads']  # 12
    # feedforward hidden size
    hid_dim_forward = config['intermediate_size']  # 3072
    # cls hidden size
    hid_dim_cls = config['pooler_fc_size']  # 768
    # reg hidden size
    hid_dim_reg = config['hidden_size']  # 768

    input1 = Input(shape=(max_len,), name='Input-Token')
    input2 = Input(shape=(max_len,), name='Input-Segment')
    input3 = Input(shape=(max_len,), name='Input-Masked')
    inputs = [input1, input2, input3]

    token_embedding_layer, token_weights = TokenEmbedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True, name='Embedding-Token')(input1)

    segment_embedding_layer = Embedding(input_dim=2, output_dim=emb_dim, name='Embedding-Segment')(input2)

    token_add_segment_layer = Add(name='Embedding-Token-Segment')([token_embedding_layer, segment_embedding_layer])

    position_embedding_layer = PositionEmbedding(input_dim=max_len, output_dim=emb_dim, name='Embedding-Position')(token_add_segment_layer)

    dropout_layer = Dropout(rate=dropout_rate, name='Embedding-Dropout')(position_embedding_layer)

    ln_layer = LayerNormalization(name='Embedding-LayerNorm')(dropout_layer)

    attention_input_layer = ln_layer

    # multihead attention block
    for i in range(block_num):
        attention_layer = MultiHeadAttention(n_head=n_head, name='MultiHeadSelfAttention-%s' % i)(attention_input_layer)
        attention_dropout_layer = Dropout(dropout_rate, name='MultiHeadSelfAttention-%s-Dropout' % i)(attention_layer)
        attention_add_layer = Add(name='MultiHeadSelfAttention-%s-Add' % i)([attention_input_layer, attention_dropout_layer])
        attention_ln_layer = LayerNormalization(name='MultiHeadSelfAttention-%s-LayerNorm' % i)(attention_add_layer)

        forward_input_layer = attention_ln_layer

        forward_layer = FeedForward(hid_dim=hid_dim_forward, name='FeedForward-%s' % i)(forward_input_layer)
        forward_dropout_ayer = Dropout(dropout_rate, name='FeedForward-%s-Dropout' % i)(forward_layer)
        forward_add_ayer = Add(name='FeedForward-%s-Add' % i)([forward_input_layer, forward_dropout_ayer])
        forward_ln_ayer = LayerNormalization(name='FeedForward-%s-LayerNorm' % i)(forward_add_ayer)

        attention_input_layer = forward_ln_ayer

    # [N, max_len, emb_dim]
    base_layer = forward_ln_ayer

    if not base:
        # reg
        if 1:
            reg_dense_layer = Dense(hid_dim_reg, name='Reg-Dense', activation='gelu')(base_layer)
            reg_ln_layer = LayerNormalization(name='Reg-LayerNorm')(reg_dense_layer)
            # [N, max_len, vocab_size]
            reg_sim_layer = EmbeddingSimilarity(name='Reg-Sim')([reg_ln_layer, token_weights])
            # [N, max_len, vocab_size]
            masked_layer = Masked(name='Reg')([reg_sim_layer, input3])
        # cls
        if 1:
            # base_layer = [N, max_len, emb_dim]
            # [N, emb_dim]
            extract_layer = Extract(name='Extract')(base_layer)
            cls_dense_layer = Dense(hid_dim_cls, name='Cls-Dense', activation='tanh')(extract_layer)
            cls_pred_layer = Dense(units=2, activation='softmax', name='Cls')(cls_dense_layer)

        model = keras.models.Model(inputs=inputs, outputs=[masked_layer, cls_pred_layer])

    else:
        # base layer
        model = keras.models.Model(inputs=inputs[:2], outputs=base_layer)

    return model


def get_weights(model,
                ckpt_file_name=ckpt_file,
                base=False,
                max_len=512):
    """

    load weights from official weights files

    """
    config = get_config()

    def ckpt_opener(ckpt_file):
        return lambda x: tf.train.load_variable(ckpt_file, x)

    loader = ckpt_opener(ckpt_file_name)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])

    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])

    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')[:min(max_len, config['max_position_embeddings']), :],
    ])

    model.get_layer(name='Embedding-LayerNorm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['num_hidden_layers']):
        model.get_layer(name='MultiHeadSelfAttention-%s' % i).set_weights([
            loader('bert/encoder/layer_%s/attention/self/query/kernel' % i),
            loader('bert/encoder/layer_%s/attention/self/query/bias' % i),
            loader('bert/encoder/layer_%s/attention/self/key/kernel' % i),
            loader('bert/encoder/layer_%s/attention/self/key/bias' % i),
            loader('bert/encoder/layer_%s/attention/self/value/kernel' % i),
            loader('bert/encoder/layer_%s/attention/self/value/bias' % i),
            loader('bert/encoder/layer_%s/attention/output/dense/kernel' % i),
            loader('bert/encoder/layer_%s/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='MultiHeadSelfAttention-%s-LayerNorm' % i).set_weights([
            loader('bert/encoder/layer_%s/attention/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%s/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='FeedForward-%s' % i).set_weights([
            loader('bert/encoder/layer_%s/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%s/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%s/output/dense/kernel' % i),
            loader('bert/encoder/layer_%s/output/dense/bias' % i),
        ])
        model.get_layer(name='FeedForward-%s-LayerNorm' % i).set_weights([
            loader('bert/encoder/layer_%s/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%s/output/LayerNorm/beta' % i),
        ])
    if not base:
        model.get_layer(name='Reg-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='Reg-LayerNorm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='Reg-Sim').set_weights([
            loader('cls/predictions/output_bias'),
        ])
        model.get_layer(name='Cls-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='Cls').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])
