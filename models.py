from keras.layers import *
import keras
import tensorflow as tf
from keras.utils import get_custom_objects
from tensorflow.python.ops.math_ops import erf, sqrt

keras.initializers.get('glorot_normal')
keras.initializers.get('zeros')


def custom_config():
    return {
        'TokenEmbedding': TokenEmbedding,
        'PositionEmbedding': PositionEmbedding,
        'MultiHeadAttention': MultiHeadAttention,
        'EmbeddingSimilarity': EmbeddingSimilarity,
        'LayerNormalization': LayerNormalization,
        'FeedForward': FeedForward,
        'Masked': Masked,
        'Extract': Extract,
        'gelu': gelu,
        # 'ScaledDotProductAttention': ScaledDotProductAttention
    }


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
    # from Google bert gule
    # y = 0.5 * x * (1.0 + tf.tanh(
    #     (
    #             np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    #     )
    # ))
    # return y


class TokenEmbedding(Embedding):
    """

    Token embedding

    """

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), (self.input_dim, self.output_dim)]

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs), K.identity(self.embeddings)]


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

        # softmax 1     ！！！这一步必须在mask之前！！！
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


'''
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 n_head,
                 activation='gelu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.n_head = n_head
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'n_head': self.n_head,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
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
        feature_dim = int(v[-1])
        if feature_dim % self.n_head != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.n_head, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.n_head),
                self._reshape_to_batches(k, self.n_head),
                self._reshape_to_batches(v, self.n_head),
            ],
            mask=[
                self._reshape_mask(q_mask, self.n_head),
                self._reshape_mask(k_mask, self.n_head),
                self._reshape_mask(v_mask, self.n_head),
            ],
        )
        y = self._reshape_from_batches(y, self.n_head)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        return y

class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.

    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None:
            e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
        a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value)
        if self.return_attention:
            return [v, a]
        return v
'''


class FeedForward(keras.layers.Layer):
    """

    前向传播
    # https://arxiv.org/pdf/1706.03762.pdf    --->>> page5-3.3
    这部分本质上是两个dense层

    """

    def __init__(self,
                 hid_dim: int,
                 **kwargs):
        """

        :param hid_dim:             隐层数值
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

        # dense层
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
        # outputs = (inputs - K.mean(inputs, -1, keepdims=True)) / K.std(inputs, axis=-1, keepdims=True)
        # outputs = self.gamma * outputs + self.beta
        # return outputs
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
        """

        把两个mask进行组合，一个是手动mask的mask, 一个padding的mask，

        """
        mask_combine = K.all([K.cast(inputs[1], bool), mask[0]], axis=0)
        return mask_combine

    def call(self, inputs, mask=None, **kwargs):
        return inputs[0]


class Extract(Layer):
    """

    features extract layer , for extracting the CLS vector generally

    """

    def __init__(self, index, **kwargs):
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


get_custom_objects().update(custom_config())
