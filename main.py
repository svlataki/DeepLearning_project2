
from src.training.trainer import Trainer
import os
from src.models.pretrained_cnn import PRETRAINED_CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
tf.random.set_seed(42)
import argparse

def parse_arguments():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--task', type=str, required=True)
    my_parser.add_argument('--tune', action='store_true')
    
    return my_parser.parse_args()

def main():

    pixels = 28*28
    num_categories = 2

        
    model = PRETRAINED_CNN(pixels,num_categories)
        
    
    model_object = model.baseline_model()

    my_trainer = Trainer()

    print(my_trainer.test_generator.class_indices.keys())
    my_trainer.train(model_object)

    predictions = my_trainer.predict(model_object)
    pd.DataFrame(predictions).to_csv('predictions2.csv',index=False)

    # print(my_trainer.val_generator.class_indices)
    # print('------------------------------------------')
    # print(my_trainer.val_generator.class_indices.keys())
    # print('------------------------------------------')
    # print(my_trainer.val_generator.classes)

    my_trainer.confusion_matrix(predictions)

    my_trainer.class_report(predictions)

if __name__ == "__main__":
    main()

