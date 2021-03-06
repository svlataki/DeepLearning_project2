
from cgi import print_arguments
from re import A
from src.training.trainer import Trainer
import os
from src.models.pretrained_cnn import PRETRAINED_CNN
from src.models.plain_cnn import CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
tf.random.set_seed(42)
import argparse

def parse_arguments():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    
    return my_parser.parse_args()

def main(model_type):

    num_categories = 1
    if model_type == 'pretrained_cnn':
        model = PRETRAINED_CNN(num_categories)
    elif model_type == 'cnn':
        model = CNN(num_categories)

    model_object = model.baseline_model()

    my_trainer = Trainer()

    my_trainer.train(model_object)

    my_trainer.plot_metrics(model_type)

    predictions = my_trainer.predict(model_object)

    my_trainer.confusion_matrix(predictions,model_type)

    my_trainer.class_report(predictions)

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments.model)

# # https://github.com/sid0407/MURA-XR_SHOULDER/blob/master/MURA(DenseNet)%20.ipynb