import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, GlobalAveragePooling1D, Dense, MultiHeadAttention
import numpy as np

# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.calculate_positional_encoding(max_position, d_model)
        
    def calculate_positional_encoding(self, max_position, d_model):
        positional_encoding = np.zeros((max_position, d_model))
        position = np.arange(0, max_position, dtype=np.float32)[:, np.newaxis]
        div_term = tf.pow(10000.0, 2 * tf.range(0, self.d_model, 2, dtype=tf.float32) / tf.cast(self.d_model, tf.float32))
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.convert_to_tensor(positional_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.Key_dim = d_model // self.num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.Key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # Linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)

        # Transpose and reshape
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear layer
        outputs = self.dense(concat_attention)

        return outputs

    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        return output

class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.001):
        super(TransformerBlock, self).__init__()
        #self.att = MultiHeadSelfAttention(d_model, num_heads)
        key_dim = d_model // num_heads
        self.att = MultiHeadAttention(num_heads,key_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation='relu'), Dense(d_model)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        #attn_output = self.att({'query': inputs, 'key': inputs, 'value': inputs, 'mask': None})
        attn_output = self.att(inputs,inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
#256#8#4 //512#9#20 
def build_transformer_model(max_len, vocab_size, d_model=512, num_heads=9, ff_dim=20, num_classes=3):
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    #positional_encoding = PositionalEncoding(vocab_size, d_model)
    #positional_layer = positional_encoding(embedding_layer)
    transformer_block = TransformerBlock(d_model, num_heads, ff_dim)(embedding_layer)
    pooling_layer = GlobalAveragePooling1D()(transformer_block)
    outputs = Dense(num_classes, activation='sigmoid')(pooling_layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
