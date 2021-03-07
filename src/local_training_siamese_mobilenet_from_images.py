import argparse
import logging
import os
import pathlib

import tarfile
# LOCAL TRAIN
#logging.basicConfig(filename='./job.log',level=logging.DEBUG)
#Sagemaker train
logging.basicConfig(filename='/opt/ml/model/job.log',level=logging.DEBUG)

import tensorflow as tf
from data.image_generators import BalancedDataGeneratorFromImages, BalancedPairsGeneratorFromImages
from models.siamese_twin_images import siamese_net_from_images_mobilenet
from models.siamese_twin_predictions import (
    predict_siamese_twin_mobilenet_model,
    write_prediction_file,
)


if __name__ == "__main__":
    logging.info("Training starting up!")

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=100)

    # input data and model directories
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        #"--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
        "--train", type=str, default='./imgs/train'
    )
    parser.add_argument(
        #"--eval", type=str, default=os.environ.get("SM_CHANNEL_EVAL")
        "--eval", type=str, default='./imgs/test-val'
    )

    args, _ = parser.parse_known_args()

    logging.info("Parsed arguments: %s", args)

    if "SM_MODEL_DIR" in os.environ.keys():
        model_dir = "/opt/ml/model/"
    else:
        model_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "trained_models",
        )

    logging.info("model_dir: %s", model_dir)
    
    #### MY MODIFICATION
    logging.info("Start decompressing images into ./imgs/")
    # Extract the tail edge images 
    tar = tarfile.open('./img_extract.tar.gz', "r:gz")
    tar.extractall(path='./imgs/')
    tar.close()
    logging.info("Decompression done")
    ####
    
    logging.info(
        #"Instantiating BalancedDataGeneratorFromImages with args: %s", args.train
        "Instantiating BalancedPairsGeneratorFromImages with args: %s", args.train
    )
    #train_data_generator = BalancedDataGeneratorFromImages(args.train)
    train_data_generator = BalancedPairsGeneratorFromImages(args.train) 
    logging.info(
        "Initializing siamese_net_from_images_mobilenet with learning_rate: %s",
        args.learning_rate,
    )
    model = siamese_net_from_images_mobilenet(args.learning_rate)
    logging.info(
        "Initializing siamese_net_from_images_mobilenet with learning_rate %s finished.",
        args.learning_rate,
    )
    model.summary()

    # Create a callback that saves the training metrics
    log_dir = os.path.join(model_dir, "tensorboard_logs")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1,
    )

    # Create a callback that saves the model's weights
    checkpoints_dir = os.path.join(model_dir, "checkpoints",)
    # Ugly, but we need to have the placeholder within the folder.
    checkpoints_dir = "%s/checkpoint{epoch:04d}.ckpt" % checkpoints_dir
    logging.info(
        "Will use log_dir: %s and checkpoints_dir: %s",
        log_dir,
        checkpoints_dir,
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoints_dir, save_weights_only=True, verbose=1,
    )

    logging.info(
        "Instantiating fit with args: steps:%s, epochs:%s",
        args.steps_per_epoch,
        args.epochs,
    )

    model.fit_generator(
        train_data_generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        #callbacks=[tensorboard_callback, checkpoint_callback],
        callbacks=[tensorboard_callback], # exclude checkpoint for smaller output file to download.
    )

    logging.info("Saving model to %s", model_dir)
    model.save(model_dir)

    logging.info(
        "Calling predict_siamese_twin_mobilenet_model with train: %s and eval: %s",
        args.train,
        args.eval,
    )
    predictions = predict_siamese_twin_mobilenet_model(
        args.train, args.eval, model
    )

    logging.info("Writing predictions")
    write_prediction_file(
        predictions, os.path.join(model_dir, "predictions.csv")
    )
