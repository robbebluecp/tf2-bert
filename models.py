from keras.layers import *
import keras
import tensorflow as tf
from keras.utils import get_custom_objects


default_weights_initializer = keras.initializers.get('glorot_normal')
default_bias_initializer = keras.initializers.get('zeros')


def custom_config():
    return {
        'TokenEmbedding': TokenEmbedding,
        'PositionEmbedding': PositionEmbedding,
        'MultiHeadAttention': MultiHeadAttention,
        'EmbeddingSimilarity': EmbeddingSimilarity,
        'LayerNormalization': LayerNormalization,
        'FeedForward': FeedForward,
        'Masked': Masked,
        'gelu': gelu
    }


def gelu(x):
    # from Google bert gule
    y = 0.5 * x * (1.0 + tf.tanh(
        (
                np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
        )
    ))
    return y


class TokenEmbedding(Embedding):
    """

    Token embedding

    """

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), (self.input_dim, self.output_dim)]

    def compute_mask(self, inputs, mask=None):
        return super(TokenEmbedding, self).compute_mask(inputs, mask)

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
        self.supports_masking = False
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
            initializer=default_weights_initializer,
            name='position_weights'
        )
        super(PositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        N = input_shape[0]

        # [N, max_len, emb_dim]
        pos_embeddings = K.tile(
            K.expand_dims(self.position_weights, axis=0),
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
            initializer=default_weights_initializer,
            name='%s_wq' % self.name,
        )
        self.bq = self.add_weight(
            shape=(emb_dim,),
            initializer=default_bias_initializer,
            name='%s_bq' % self.name,
        )
        self.wk = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=default_weights_initializer,
            name='%s_wk' % self.name,
        )
        self.bk = self.add_weight(
            shape=(emb_dim,),
            initializer=default_bias_initializer,
            name='%s_bk' % self.name,
        )
        self.wv = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=default_weights_initializer,
            name='%s_wv' % self.name
        )
        self.bv = self.add_weight(
            shape=(emb_dim,),
            initializer=default_bias_initializer,
            name='%s_bv' % self.name
        )
        self.wo = self.add_weight(
            shape=(emb_dim, emb_dim),
            initializer=default_weights_initializer,
            name='%s_wo' % self.name,
        )
        self.bo = self.add_weight(
            shape=(emb_dim,),
            initializer=default_bias_initializer,
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
        x = tf.split(x, n_head, -1)
        x = K.concatenate(x, axis=0)
        return x

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
        self.kernel_initializer = keras.initializers.get(default_weights_initializer)
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
        outputs = (inputs - K.mean(inputs, -1, keepdims=True)) / K.std(inputs, axis=-1, keepdims=True)
        outputs = self.gamma * outputs + self.beta
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
