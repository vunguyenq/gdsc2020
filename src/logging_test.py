import argparse
import logging
import os
import pathlib

import tarfile
logging.basicConfig(filename='job.log',level=logging.DEBUG)

#logging.basicConfig(filename='/opt/ml/model/job.log',level=logging.DEBUG)
#import tensorflow as tf
#from data.image_generators import BalancedDataGeneratorFromImages
#from models.siamese_twin_images import siamese_net_from_images_mobilenet
#from models.siamese_twin_predictions import (
#    predict_siamese_twin_mobilenet_model,
#    write_prediction_file,
#)


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
        "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--eval", type=str, default=os.environ.get("SM_CHANNEL_EVAL")
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

    logging.info(
        "Instantiating BalancedDataGeneratorFromImages with args: %s", args.train
    )

    logging.info(
        "Initializing siamese_net_from_images_mobilenet with learning_rate: %s",
        args.learning_rate,
    )

    logging.info(
        "Initializing siamese_net_from_images_mobilenet with learning_rate %s finished.",
        args.learning_rate,
    )

    # Create a callback that saves the training metrics
    log_dir = os.path.join(model_dir, "tensorboard_logs")
    
    # Create a callback that saves the model's weights
    checkpoints_dir = os.path.join(model_dir, "checkpoints",)
    # Ugly, but we need to have the placeholder within the folder.
    checkpoints_dir = "%s/checkpoint{epoch:04d}.ckpt" % checkpoints_dir

    logging.info(
        "Will use log_dir: %s and checkpoints_dir: %s",
        log_dir,
        checkpoints_dir,
    )


    logging.info(
        "Instantiating fit with args: steps:%s, epochs:%s",
        args.steps_per_epoch,
        args.epochs,
    )

    logging.info("Saving model to %s", model_dir)
    
    logging.info(
        "Calling predict_siamese_twin_mobilenet_model with train: %s and eval: %s",
        args.train,
        args.eval,
    )
    
    logging.info("Writing predictions")
    
    logging.info("Start decompressing images into ./imgs/")
    # Extract the tail edge images 
    #tar = tarfile.open('./img_extract.tar.gz', "r:gz")
    #tar.extractall(path='./imgs/')
    #tar.close()
    #logging.info("Decompression done")
    
    # Examine extracted files:
    inputPath = pathlib.Path('./imgs/')
    folders = list(inputPath.glob('*'))
    eval_folder = folders[0]
    train_folder = folders[1]
    eval_file1 = list(eval_folder.glob('*'))[0]
    eval_file_count = len(list(eval_folder.glob('*')))
    whales = list(train_folder.glob('*'))
    whale_count = len(whales)
    train_file_count = 0
    train_file1 = list(whales[0].glob('*'))[0]
    for whale in whales:
        train_file_count += len(list(whale.glob('*')))
    #print(eval_folder, eval_file1, eval_file_count, train_folder, len(whales),train_file_count)
    logging.info(
        "***File statistics*** Eval folder: %s | Sample eval file: %s | Eval file count: %s | Train folder: %s | Whale count: %s | Train file count: %s | Sample train file: %s ",
        eval_folder, eval_file1, eval_file_count, train_folder, whale_count,train_file_count, train_file1
    )
    
    import traceback
    try:
        1/0 
    except Exception as e:
        with open('log.txt', 'a') as f:
            f.write(str(e))
            f.write(traceback.format_exc())