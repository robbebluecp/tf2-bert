from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.ops.math_ops import erf, sqrt
import json
import numpy as np
import tensorflow_addons as tfa
import config


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


class TokenEmbedding(Embedding):
    """

    Token embedding

    """

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), (self.input_dim, self.output_dim)]

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs), tf.identity(self.embeddings)]


class PositionEmbedding(Layer):
    """

    Position embedding

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 **kwargs):
        """

        :param input_dim:           max_len(512)
        :param output_dim:          emb_dim(768)
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.supports_masking = True
        super(PositionEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  }
        base_config = super(PositionEmbedding, self).get_config()
        base_config.update(config)
        return base_config

    def build(self, input_shape):
        self.position_weights = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='uniform',
            name='position_weights'
        )
        super(PositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        N, max_len = input_shape[0], input_shape[1]

        # [N, max_len, emb_dim]
        pos_embeddings = K.tile(
            K.expand_dims(self.position_weights[:max_len, :], axis=0),
            [N, 1, 1],
        )
        return inputs + pos_embeddings


class MultiHeadAttention(Layer):
    """

    Multi-head attention layer

    """

    def __init__(self,
                 n_head: int,
                 **kwargs):
        """

        :param n_head:      head num

        """
        self.supports_masking = True
        self.n_head = n_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'n_head': self.n_head
        }
        base_config = super(MultiHeadAttention, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q = input_shape[0]
            return K.int_shape(q)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        emb_dim = v[-1]
        self.wq = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=keras.initializers.get('glorot_normal'),
            name='%s_wq' % self.name,
        )
        self.bq = self.add_weight(
            shape=(emb_dim,),
            initializer=keras.initializers.get('zeros'),
            name='%s_bq' % self.name,
        )
        self.wk = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=keras.initializers.get('glorot_normal'),
            name='%s_wk' % self.name,
        )
        self.bk = self.add_weight(
            shape=(emb_dim,),
            initializer=keras.initializers.get('zeros'),
            name='%s_bk' % self.name,
        )
        self.wv = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=keras.initializers.get('glorot_normal'),
            name='%s_wv' % self.name
        )
        self.bv = self.add_weight(
            shape=(emb_dim,),
            initializer=keras.initializers.get('zeros'),
            name='%s_bv' % self.name
        )
        self.wo = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=keras.initializers.get('glorot_normal'),
            name='%s_wo' % self.name,
        )
        self.bo = self.add_weight(
            shape=(emb_dim,),
            initializer=keras.initializers.get('zeros'),
            name='%s_bo' % self.name,
        )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):

        q = k = v = inputs

        q_mask = k_mask = v_mask = mask

        # [N, max_len, emb_dim] * [emb_dim, emb_dim] = [N, max_len, emb_dim]
        q = K.dot(q, self.wq)
        k = K.dot(k, self.wk)
        v = K.dot(v, self.wv)

        q += self.bq
        k += self.bk
        v += self.bv

        # scale dot product
        y = ScaleDotProducttion()([q, k, v], [q_mask, k_mask, v_mask], self.n_head)

        y = ScaleDotProducttion.reshape_from_attention_shape(y, self.n_head)
        y = K.dot(y, self.wo)
        y += self.bo

        return y


class ScaleDotProducttion:
    """

    scale dot product
    https://arxiv.org/pdf/1706.03762.pdf        page4-3.2.1

    """

    def call(self, inputs, masks, n_head):
        q, k, v = inputs
        q = self.reshape_to_attention_shape(q, n_head)
        k = self.reshape_to_attention_shape(k, n_head)
        v = self.reshape_to_attention_shape(v, n_head)

        # every mask is the same
        mask = masks[0]

        emb_dim = K.shape(q)[-1]

        # [N * n_head, max_len, max_len]
        scores = K.batch_dot(q, k, axes=2) / K.sqrt(K.cast(emb_dim, K.floatx()))

        # softmax 1
        scores = K.exp(scores - K.max(scores, axis=-1, keepdims=True))

        if mask is not None:
            mask = self.reshape_mask(mask, n_head)
            # [N * n_head, max_len, max_len] * [N * n_head, 1, max_len]
            scores *= mask

        # softmax 2
        scores /= (K.sum(scores, axis=-1, keepdims=True) + K.epsilon())

        # [N * n_head, max_len, emb_dim]
        y = K.batch_dot(scores, v)
        return y

    @staticmethod
    def reshape_to_attention_shape(x,
                                   n_head: int):

        """

        from raw shape([N, max_len, emb_dim]) reshape to mutil head shape ([N*n_head, max_len, emb_dim / n_head])

        """
        input_shape = K.shape(x)
        N, max_len, emb_dim = input_shape[0], input_shape[1], input_shape[2]
        x = tf.split(x, n_head, -1)
        x = K.concatenate(x, axis=0)
        x = K.reshape(x, (n_head, N, max_len, emb_dim // n_head))
        x = K.permute_dimensions(x, (1, 0, 2, 3))
        y = K.reshape(x, (-1, max_len, emb_dim // n_head))
        return y

    @staticmethod
    def reshape_from_attention_shape(x,
                                     n_head: int):
        """

        reshape back to raw shape from multi head shape

        """
        input_shape = K.shape(x)
        N, max_len, emb_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (N // n_head, n_head, max_len, emb_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (N // n_head, max_len, emb_dim * n_head))

    @staticmethod
    def reshape_mask(mask,
                     n_head: int):
        """

        reshape mask

        """
        if mask is None:
            return mask
        max_len = K.shape(mask)[1]
        mask = K.tile(mask, (1, n_head))
        mask = K.reshape(mask, (-1, max_len))
        mask = K.expand_dims(mask, 1)
        return K.cast(mask, K.floatx())

    def __call__(self, inputs, masks, head_num, *args, **kwargs):
        return self.call(inputs, masks, head_num)


class FeedForward(keras.layers.Layer):
    """

    # https://arxiv.org/pdf/1706.03762.pdf

    """

    def __init__(self,
                 hid_dim: int,
                 **kwargs):
        """

        :param hid_dim:
        :param dropout_rate:        dropout rate

        """
        self.supports_masking = True
        self.hid_dim = hid_dim
        self.activation = keras.activations.get('gelu')
        self.kernel_initializer = keras.initializers.get('glorot_normal')
        self.bias_initializer = keras.initializers.get('zeros')
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'hid_dim': self.hid_dim,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        emb_dim = input_shape[-1]
        # [emb_dim, emb_dim]
        self.w1 = self.add_weight(
            shape=(emb_dim, self.hid_dim),
            initializer=self.kernel_initializer,
            name='%s_w1' % self.name,
        )

        # [emb_dim, ]
        self.b1 = self.add_weight(
            shape=(self.hid_dim,),
            initializer=self.bias_initializer,
            name='%s_b1' % self.name,
        )
        self.w2 = self.add_weight(
            shape=(self.hid_dim, emb_dim),
            initializer=self.kernel_initializer,
            name='%s_w2' % self.name,
        )
        self.b2 = self.add_weight(
            shape=(emb_dim,),
            initializer=self.bias_initializer,
            name='%s_b2' % self.name,
        )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None):
        # [N, max_len, emb_dim]

        # dense
        h = K.dot(x, self.w1)
        h = K.bias_add(h, self.b1)
        h = self.activation(h)

        # dropout

        # [N, max_len, emb_dim]
        y = K.dot(h, self.w2)
        y = K.bias_add(y, self.b2)
        return y


class LayerNormalization(Layer):
    """

    Layer Norm
    https://arxiv.org/pdf/1607.06450.pdf

    gamma * (X - mean) / std + beta

    during per layer(or batch you can consider like it)

    """

    def __init__(self,
                 **kwargs):
        self.supports_masking = True
        self.epsilon = pow(K.epsilon(), 2)
        super(LayerNormalization, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        emb_dim = input_shape[-1]
        self.gamma = self.add_weight(
            shape=(emb_dim,),
            initializer=keras.initializers.get('ones'),
            name='gamma',
        )
        self.beta = self.add_weight(
            shape=(emb_dim,),
            initializer=keras.initializers.get('zeros'),
            name='beta',
        )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, mask=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs *= self.gamma
        outputs += self.beta
        return outputs


class EmbeddingSimilarity(Layer):

    def __init__(self,
                 **kwargs):
        super(EmbeddingSimilarity, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        vocab_size = input_shape[-1][0]
        self.bias = self.add_weight(
            shape=(vocab_size,),
            initializer=keras.initializers.get('zeros'),
            name='bias',
        )
        super(EmbeddingSimilarity, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # [N, max_Len, vocab_size]
        return input_shape[0][:-1] + (input_shape[-1][0],)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        # [N, max_len, emb_dim] [vacab_size, emb_dim]
        inputs, embeddings = inputs
        outputs = K.bias_add(K.dot(inputs, K.transpose(embeddings)), self.bias)
        return keras.activations.softmax(outputs)


class Masked(Layer):
    """

    combine mask layer

    """

    def __init__(self,
                 **kwargs):
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        # [N, max_len, vocab_size]  [max_len, ]
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        mask_combine = K.all([K.cast(inputs[1], bool), mask[0]], axis=0)
        return mask_combine

    def call(self, inputs, mask=None, **kwargs):
        return inputs[0]


class Extract(Layer):
    """

    features extract layer , for extracting the CLS vector generally, in fact this dimension of vector
    also means a sentence vector


    """

    def __init__(self, index=0, **kwargs):
        super(Extract, self).__init__(**kwargs)
        self.index = index
        self.supports_masking = True

    def get_config(self):
        config = {
            'index': self.index,
        }
        base_config = super(Extract, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        # [N, emb_dim]
        return input_shape[:1] + input_shape[2:]

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # [N, emd_dim]
        return x[:, self.index]

class CRF(tf.keras.layers.Layer):
    """
        Conditional Random Field layer (tf.keras)
        `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
        must be equal to the number of classes the CRF can predict (a linear layer is recommended).
        Note: the loss and accuracy functions of networks using `CRF` must
        use the provided loss and accuracy functions (denoted as loss and viterbi_accuracy)
        as the classification of sequences are used with the layers internal weights.
        Args:
            output_dim (int): the number of labels to tag each temporal input.
        Input shape:
            nD tensor with shape `(batch_size, sentence length, num_classes)`.
        Output shape:
            nD tensor with shape: `(batch_size, sentence length, num_classes)`.
        """

    def __init__(self,
                 output_dim,
                 mode='reg',
                 supports_masking=False,
                 transitions=None,
                 **kwargs):
        self.transitions = None
        super(CRF, self).__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.mode = mode
        if self.mode == 'pad':
            self.input_spec = [tf.keras.layers.InputSpec(min_ndim=3), tf.keras.layers.InputSpec(min_ndim=2)]
        elif self.mode == 'reg':
            self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        else:
            raise ValueError
        self.supports_masking = supports_masking
        self.sequence_lengths = None

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'mode': self.mode,
            'supports_masking': self.supports_masking,
            'transitions': tf.keras.backend.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == 'pad':
            assert len(input_shape) == 2
            assert len(input_shape[0]) == 3
            assert len(input_shape[1]) == 2
            f_shape = tf.TensorShape(input_shape[0])
            input_spec = [tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]}),
                          tf.keras.layers.InputSpec(min_ndim=2, axes={-1: 1}, dtype=tf.int32)]
        else:
            assert len(input_shape) == 3
            f_shape = tf.TensorShape(input_shape)
            input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output shape. '
                             'Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        if self.mode == 'pad':
            sequences = tf.convert_to_tensor(inputs[0], dtype=self.dtype)
            self.sequence_lengths = tf.keras.backend.flatten(inputs[-1])
        else:
            sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
            shape = tf.shape(inputs)
            self.sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
        viterbi_sequence, _ = tfa.text.crf.crf_decode(sequences, self.transitions,
                                                        self.sequence_lengths)
        output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
        return tf.keras.backend.in_train_phase(sequences, output)

    def loss(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        log_likelihood, self.transitions = tfa.text.crf.crf_log_likelihood(y_pred,
                                                                             tf.cast(tf.keras.backend.argmax(y_true),
                                                                                     dtype=tf.int32),
                                                                             self.sequence_lengths,
                                                                             transition_params=self.transitions)
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        if self.mode == 'pad':
            data_shape = input_shape[0]
        else:
            data_shape = input_shape
        tf.TensorShape(data_shape).assert_has_rank(3)
        return data_shape[:2] + (self.output_dim,)

    @property
    def viterbi_accuracy(self):
        def accuracy(y_true, y_pred):
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            viterbi_sequence, _ = tfa.text.crf.crf_decode(y_pred, self.transitions, sequence_lengths)
            output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
            return tf.keras.metrics.categorical_accuracy(y_true, output)

        accuracy.func_name = 'viterbi_accuracy'
        return accuracy


custom_config = {
    'TokenEmbedding': TokenEmbedding,
    'PositionEmbedding': PositionEmbedding,
    'MultiHeadAttention': MultiHeadAttention,
    'EmbeddingSimilarity': EmbeddingSimilarity,
    'LayerNormalization': LayerNormalization,
    'FeedForward': FeedForward,
    'Masked': Masked,
    'Extract': Extract,
    'gelu': gelu,
    'CRF': CRF,
    # 'ScaledDotProductAttention': ScaledDotProductAttention
}

get_custom_objects().update(custom_config)


class BERT:

    def __init__(self, 
                 bert_file_path: str = config.bert_path,
                 base: bool or int = False,
                 max_len: int = 512,
                 load_pre: bool = True
                 ):
        self.bert_file_path = bert_file_path
        self.conf_file = self.bert_file_path + '/bert_config.json'
        self.ckpt_file = self.bert_file_path + '/bert_model.ckpt'
        self.config = self.get_config(self.conf_file)
        self.base = base
        self.max_len = max_len
        self.load_pre = load_pre

    def __call__(self, *args, **kwargs):
        self.model = self.load_model()
        if self.load_pre:
            self.load_weights()
        return self.model

    @staticmethod
    def get_config(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            f.close()
        return config

    @staticmethod
    def ckpt_opener(ckpt_file):
        return lambda x: tf.train.load_variable(ckpt_file, x)

    def load_model(self):
        # embedding vocab size
        vocab_size = self.config['vocab_size']  # 21128
        # max length each sentence
        max_len = min(self.config['max_position_embeddings'], self.max_len)  # 512
        # embedding output_dim
        emb_dim = self.config['hidden_size']  # 768
        #
        dropout_rate = self.config['hidden_dropout_prob']  # 0.1
        # attention block num
        block_num = self.config['num_hidden_layers']  # 12
        # head num
        n_head = self.config['num_attention_heads']  # 12
        # feedforward hidden size
        hid_dim_forward = self.config['intermediate_size']  # 3072
        # cls hidden size
        hid_dim_cls = self.config['pooler_fc_size']  # 768
        # reg hidden size
        hid_dim_reg = self.config['hidden_size']  # 768

        input1 = Input(shape=(max_len,), name='Input-Token')
        input2 = Input(shape=(max_len,), name='Input-Segment')
        input3 = Input(shape=(max_len,), name='Input-Masked')
        inputs = [input1, input2, input3]

        token_embedding_layer, token_weights = TokenEmbedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True,
                                                              name='Embedding-Token')(input1)

        segment_embedding_layer = Embedding(input_dim=2, output_dim=emb_dim, name='Embedding-Segment')(input2)

        token_add_segment_layer = Add(name='Embedding-Token-Segment')([token_embedding_layer, segment_embedding_layer])

        position_embedding_layer = PositionEmbedding(input_dim=max_len, output_dim=emb_dim, name='Embedding-Position')(
            token_add_segment_layer)

        dropout_layer = Dropout(rate=dropout_rate, name='Embedding-Dropout')(position_embedding_layer)

        ln_layer = LayerNormalization(name='Embedding-LayerNorm')(dropout_layer)

        attention_input_layer = ln_layer

        # multihead attention block
        for i in range(block_num):
            attention_layer = MultiHeadAttention(n_head=n_head, name='MultiHeadSelfAttention-%s' % i)(
                attention_input_layer)
            attention_dropout_layer = Dropout(dropout_rate, name='MultiHeadSelfAttention-%s-Dropout' % i)(
                attention_layer)
            attention_add_layer = Add(name='MultiHeadSelfAttention-%s-Add' % i)(
                [attention_input_layer, attention_dropout_layer])
            attention_ln_layer = LayerNormalization(name='MultiHeadSelfAttention-%s-LayerNorm' % i)(attention_add_layer)

            forward_input_layer = attention_ln_layer

            forward_layer = FeedForward(hid_dim=hid_dim_forward, name='FeedForward-%s' % i)(forward_input_layer)
            forward_dropout_ayer = Dropout(dropout_rate, name='FeedForward-%s-Dropout' % i)(forward_layer)
            forward_add_ayer = Add(name='FeedForward-%s-Add' % i)([forward_input_layer, forward_dropout_ayer])
            forward_ln_ayer = LayerNormalization(name='FeedForward-%s-LayerNorm' % i)(forward_add_ayer)

            attention_input_layer = forward_ln_ayer

        # [N, max_len=512, emb_dim=768]
        base_layer = forward_ln_ayer
        if not self.base:
            # reg
            if 1:
                reg_dense_layer = Dense(hid_dim_reg, name='Reg-Dense', activation='gelu')(base_layer)
                reg_ln_layer = LayerNormalization(name='Reg-LayerNorm')(reg_dense_layer)
                # [N, max_len, emb_dim]  * [emb_dim, vocab_size]
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
    
    def load_weights(self):

        loader = BERT.ckpt_opener(self.ckpt_file)

        self.model.get_layer(name='Embedding-Token').set_weights([
            loader('bert/embeddings/word_embeddings'),
        ])

        self.model.get_layer(name='Embedding-Segment').set_weights([
            loader('bert/embeddings/token_type_embeddings'),
        ])

        self.model.get_layer(name='Embedding-Position').set_weights([
            loader('bert/embeddings/position_embeddings')[:min(self.max_len, self.config['max_position_embeddings']), :],
        ])

        self.model.get_layer(name='Embedding-LayerNorm').set_weights([
            loader('bert/embeddings/LayerNorm/gamma'),
            loader('bert/embeddings/LayerNorm/beta'),
        ])
        for i in range(self.config['num_hidden_layers']):
            self.model.get_layer(name='MultiHeadSelfAttention-%s' % i).set_weights([
                loader('bert/encoder/layer_%s/attention/self/query/kernel' % i),
                loader('bert/encoder/layer_%s/attention/self/query/bias' % i),
                loader('bert/encoder/layer_%s/attention/self/key/kernel' % i),
                loader('bert/encoder/layer_%s/attention/self/key/bias' % i),
                loader('bert/encoder/layer_%s/attention/self/value/kernel' % i),
                loader('bert/encoder/layer_%s/attention/self/value/bias' % i),
                loader('bert/encoder/layer_%s/attention/output/dense/kernel' % i),
                loader('bert/encoder/layer_%s/attention/output/dense/bias' % i),
            ])
            self.model.get_layer(name='MultiHeadSelfAttention-%s-LayerNorm' % i).set_weights([
                loader('bert/encoder/layer_%s/attention/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%s/attention/output/LayerNorm/beta' % i),
            ])
            self.model.get_layer(name='FeedForward-%s' % i).set_weights([
                loader('bert/encoder/layer_%s/intermediate/dense/kernel' % i),
                loader('bert/encoder/layer_%s/intermediate/dense/bias' % i),
                loader('bert/encoder/layer_%s/output/dense/kernel' % i),
                loader('bert/encoder/layer_%s/output/dense/bias' % i),
            ])
            self.model.get_layer(name='FeedForward-%s-LayerNorm' % i).set_weights([
                loader('bert/encoder/layer_%s/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%s/output/LayerNorm/beta' % i),
            ])
        if not self.base:
            try:
                self.model.get_layer(name='Reg-Dense').set_weights([
                    loader('cls/predictions/transform/dense/kernel'),
                    loader('cls/predictions/transform/dense/bias'),
                ])
            except:
                print('Lack of Reg-Dense:', 'Pass')

            try:
                self.model.get_layer(name='Reg-LayerNorm').set_weights([
                    loader('cls/predictions/transform/LayerNorm/gamma'),
                    loader('cls/predictions/transform/LayerNorm/beta'),
                ])
            except:
                print('Lack of Reg-LayerNorm:', 'Pass')
            try:
                self.model.get_layer(name='Reg-Sim').set_weights([
                    loader('cls/predictions/output_bias'),
                ])
            except:
                print('Lack of Reg-Sim:', 'Pass')
            try:
                self.model.get_layer(name='Cls-Dense').set_weights([
                    loader('bert/pooler/dense/kernel'),
                    loader('bert/pooler/dense/bias'),
                ])
            except:
                print('Lack of Cls-Dense:', 'Pass')
            try:
                self.model.get_layer(name='Cls').set_weights([
                    np.transpose(loader('cls/seq_relationship/output_weights')),
                    loader('cls/seq_relationship/output_bias'),
                ])
            except:
                print('Lack of Cls:', 'Pass')
