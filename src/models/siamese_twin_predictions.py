import logging
from pathlib import Path
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from data.image_generators import (
    PredictGeneratorFromImages,
    IMG_HEIGHT,
    IMG_WIDTH,
)
from data.embedding_generators import (
    DataGeneratorFromEmbeddings,
    EMBEDDING_SHAPE,
)


def predict_siamese_twin_mobilenet_model(
    train_data_path: str, test_val_data_path: str, model
) -> list:
    """
    Predicts top 20 similar images for test_val_path using the trained model
    :param train_data_path: path to train data
    :param test_val_data_path: path to test_val data
    :param model: trained model
    :return: list with best predictions
    """
    test_files_paths = list(Path(test_val_data_path).glob("*.jpg"))
    train_files_paths = list(Path(train_data_path).glob("*/*.jpg"))
    all_paths = np.concatenate((test_files_paths, train_files_paths))

    # Predict and cache the embeddings
    logging.info("Caching embeddings")
    image_generator = PredictGeneratorFromImages(
        all_paths, dim=(IMG_HEIGHT, IMG_WIDTH), shuffle=False, batch_size=32
    )
    cnn = model.layers[2]
    embeddings = cnn.predict_generator(image_generator)
    embeddings = embeddings[: len(all_paths)]
    assert len(all_paths) == len(embeddings)
    all_embeddings = dict(zip(all_paths, embeddings))

    # Rebuild the model for predicting similarities
    logging.info("Building model.")
    embeddings_1 = Input([EMBEDDING_SHAPE])
    embeddings_2 = Input([EMBEDDING_SHAPE])
    x = model.layers[3]([embeddings_1, embeddings_2])
    x = model.layers[4](x)
    output = model.layers[5](x)
    model_from_embeddings = Model(
        inputs=[embeddings_1, embeddings_2], outputs=output
    )

    # Predict the similarities
    logging.info("Starting Predictions.")
    siamese_distance_only_preds = []
    for test_file_path in test_files_paths:
        mask = all_paths != test_file_path
        all_other_images = all_paths[mask]
        assert len(all_other_images) == (len(all_paths) - 1)
        test_file_pairs = [
            (test_file_path, other_image) for other_image in all_other_images
        ]

        embedding_generator = DataGeneratorFromEmbeddings(
            test_file_pairs, all_embeddings, labels=None, shuffle=False
        )
        predictions = model_from_embeddings.predict_generator(
            embedding_generator
        )
        predictions = predictions[: len(test_file_pairs)]
        indices = np.argsort(-predictions, axis=0)
        paths_of_top_predictions = all_other_images[indices[:20].reshape(-1)]
        names_of_top_predictions = [
            path.name for path in paths_of_top_predictions
        ]
        siamese_distance_only_preds.append(
            np.concatenate(([test_file_path.name], names_of_top_predictions))
        )
    logging.info("Predictions done.")
    return siamese_distance_only_preds


def write_prediction_file(predictions: list, file_path: str):
    """
    Writes predictions into a csv file
    :param predictions: list of predictions
    :param file_path: path to save csv file
    :return:
    """
    with open(file_path, "w") as f:
        for entry in predictions:
            preds = ",".join(entry)
            f.write(preds + "\n")
    logging.info("File in %s succesfully written.", file_path)
