import tensorflow as tf
from data.embedding_generators import EMBEDDING_SHAPE
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def siamese_net_from_embeddings(learning_rate, embed_length=EMBEDDING_SHAPE):
    """
    Model architecture based on the one provided in:
    http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    input_shape = [embed_length]
    embeddings_1 = Input(input_shape)
    embeddings_2 = Input(input_shape)

    # Add a customized layer to compute the absolute difference between the encodings
    absolute_difference = tf.math.abs(embeddings_1 - embeddings_2)

    # Add a dense layer with a sigmoid unit to generate the similarity score
    similarity_score = Dense(
        1,
        activation="sigmoid",
        bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None),
    )(absolute_difference)

    siamese_net = Model(
        inputs=[embeddings_1, embeddings_2], outputs=similarity_score
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    siamese_net.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return siamese_net
