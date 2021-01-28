import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# BATCH SIZE
B = 1
# MAXIMUM WORDS PER SENTENCE
T_w = 6
# WORD EMBEDDING
D_w = 8
# SENTENCE EMBEDDING
D_s = 1
# HEADS
h_w = 4

print("d_k_w", int(D_w / h_w))

# INPUT TENSOR
X_w = tf.Variable(np.ones((B, T_w, D_w)), trainable=False, dtype=tf.float32)

def BERT(X, h, n_enc):

    def selfAttention(X, B, T, D, d_k, n_enc, n_head):

        with tf.variable_scope(str(n_enc) + "_" + "_attention_head_" + str(n_head)):
            W_Q = tf.get_variable("W_Q",
                                  shape=[D, d_k],
                                  initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                  trainable=True,
                                  )
            W_K = tf.get_variable("W_K",
                                  shape=[D, d_k],
                                  initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                  trainable=True,
                                  )
            W_V = tf.get_variable("W_V",
                                  shape=[D, d_k],
                                  initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                  trainable=True,
                                  )

        X = tf.reshape(X, shape=[B * T, D])

        Q = tf.matmul(X, W_Q)
        K = tf.matmul(X, W_K)
        V = tf.matmul(X, W_V)

        Q = tf.reshape(Q, shape=(B, T, d_k))
        K = tf.reshape(K, shape=(B, T, d_k))
        V = tf.reshape(V, shape=(B, T, d_k))

        Z = tf.multiply(tf.matmul(Q, K, transpose_b=True), 1.0 / tf.math.sqrt(float(d_k)))
        Z = tf.nn.softmax(Z, axis=2)
        Z = tf.matmul(Z, V)

        return Z

    def multiHeadAttention(X, h, n_enc):

        B, T, D = X.get_shape()
        d_k = int(int(D) / h)

        Z = []
        for i in range(h):
            Z_i = selfAttention(X, B, T, D, d_k, i, n_enc)
            Z.append(Z_i)

        Z = tf.concat(Z, axis=2)

        with tf.variable_scope("W0_" + str(n_enc)):
            W_0 = tf.get_variable("W_0",
                              shape=[D, D],
                              initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                              trainable=True,
                              )
        Z = tf.reshape(Z, shape=(B * T, D))
        Z = tf.matmul(Z, W_0)
        Z = tf.reshape(Z, shape=(B, T, D))

        return Z

    def add_and_normalize_1(X, Z, n_enc, eps=1e-4):

        B, T, D = X.get_shape()

        with tf.variable_scope("add_and_normalize_1_" + str(n_enc)):
            beta = tf.get_variable("beta_1",
                                   shape=[D],
                                   initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                   trainable=True,
                                   )
            gamma = tf.get_variable("gamma_1",
                                    shape=[D],
                                    initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                    trainable=True,
                                    )
        Y = tf.add(X, Z)
        mean, variance = tf.nn.moments(Y, axes=[2], keep_dims=True)
        normalized_Y = (Y - mean) / tf.sqrt(variance + eps)
        Y_out = normalized_Y * gamma + beta

        return Y_out

    def add_and_normalize_2(X, Z, n_enc, eps=1e-4):

        B, T, D = X.get_shape()

        with tf.variable_scope("add_and_normalize_2_" + str(n_enc)):
            beta = tf.get_variable("beta_2",
                                   shape=[D],
                                   initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                   trainable=True,
                                   )
            gamma = tf.get_variable("gamma_2",
                                    shape=[D],
                                    initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
                                    trainable=True,
                                    )
        Y = tf.add(X, Z)
        mean, variance = tf.nn.moments(Y, axes=[2], keep_dims=True)
        normalized_Y = (Y - mean) / tf.sqrt(variance + eps)
        Y_out = normalized_Y * gamma + beta

        return Y_out

    def feed_forward_network(Y, D_h, D_out, n_enc):

        B, T, D = Y.get_shape()
        Y = tf.reshape(Y, shape=[B * T, D])

        with tf.variable_scope("feed_forward_network_" + str(n_enc)):
            W1 = tf.get_variable('W1',
                                 shape=[D, D_h],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 )
            b1 = tf.get_variable('b1',
                                 shape=[D_h],
                                 initializer=tf.zeros_initializer(),
                                 )
            W2 = tf.get_variable('W2',
                                 shape=[D_h, D_out],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 )
            b2 = tf.get_variable('b2',
                                 shape=[D_out],
                                 initializer=tf.zeros_initializer(),
                                 )

        out = tf.nn.xw_plus_b(Y, W1, b1)
        out = tf.nn.leaky_relu(out, 0.01)

        out = tf.nn.xw_plus_b(out, W2, b2)
        out = tf.reshape(out, shape=[B, T, D_out])

        return out

    # SELF-ATTENTION LAYER
    Z1 = multiHeadAttention(X, h, n_enc)

    # FIRST ADD AND NORMALIZE LAYER
    N = add_and_normalize_1(X, Z1, n_enc)

    # FEEDFORWARD NETZWORK LAYER
    Z2 = feed_forward_network(N, 2 * D_w, D_w, n_enc)

    # SECOND ADD AND NORMALIZE LAYER
    Y = add_and_normalize_2(N, Z2, n_enc)

    return Y

def compression(Y, D_out):
    T = Y.get_shape()[1]
    with tf.variable_scope("compression"):
        W = tf.get_variable('W',
                             shape=[T, D_out],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             )
        b = tf.get_variable('b',
                            shape=[D_out],
                            initializer=tf.zeros_initializer(),
                            )

        out = tf.nn.xw_plus_b(Y, W, b)
        out = tf.nn.tanh(out)

    return out

# BERT
# [B * T_w, D_w] --> [B, T_w, D_w]
X_w = tf.reshape(X_w, shape=[B, T_w, D_w])
Y_w = BERT(X_w, h_w, 0)

# COMPRESSION
# shape = [B, T_w, 1] --> [B, T_w]
Y_w = tf.squeeze(Y_w)
# shape = [B, T_w] --> [B, D_s]
X_s = compression(Y_w, D_s)
print(X_s.get_shape())

# start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())