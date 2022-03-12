
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
    # my_parser.add_argument('--task', type=str, required=True)
    # my_parser.add_argument('--tune', action='store_true')
    
    return my_parser.parse_args()

def main(model):

    pixels = 28*28
    num_categories = 2
    if model == 'pretrained_cnn':
        
        model = PRETRAINED_CNN(pixels,num_categories)

    elif model == 'pretrained_cnn':
        model = CNN(pixels,num_categories)


    
    model_object = model.baseline_model()

    my_trainer = Trainer()

    my_trainer.train(model_object)

    #my_trainer.plot_metrics()

    predictions = my_trainer.predict(model_object)

    my_trainer.confusion_matrix(predictions)

    my_trainer.class_report(predictions)

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments.model)

# # https://github.com/sid0407/MURA-XR_SHOULDER/blob/master/MURA(DenseNet)%20.ipynb