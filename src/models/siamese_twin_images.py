import tensorflow as tf
from data.image_generators import IMG_HEIGHT, IMG_WIDTH
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Input,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def siamese_net_from_images_mobilenet(
    learning_rate: float, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
):
    """
    Model architecture based on the one provided in:
    http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    # Define the tensors for the two input images
    left_input = Input(input_shape, name="input_1")
    right_input = Input(input_shape, name="input_2")

    model = MobileNet(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )
    model.trainable = True

    # Generate the encodings (feature vectors) for the two images
    embeddings_1 = model(left_input)
    embeddings_2 = model(right_input)
    # Add a customized layer to compute the absolute difference between the encodings
    absolute_difference = tf.math.abs(
        embeddings_1 - embeddings_2, name="absolute_difference"
    )
    prediction = Dense(
        1,
        activation="sigmoid",
        bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None),
        name="output",
    )(absolute_difference)
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # compile the model
    optimizer = Adam(learning_rate)
    siamese_net.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return siamese_net
