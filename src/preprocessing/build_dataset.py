

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from src.preprocessing.utils import generate_df
import hub
import os
import pandas as pd
from pathlib import Path
tf.random.set_seed(42)
np.random.seed(0)

class DataBuilder():

    def get_data(self):
        
        # load dataset
        
        dataset_root = Path(r'C:\Users\svlataki\Downloads\MURA-v1.1')
        #Dividing the image data generated into train set and validation set
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255)
        #Creating training set
        train_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'train_image_paths.csv'),
                                                directory=dataset_root.parent,
                                                target_size=(224, 224),
                                                shuffle = False,
                                                batch_size=512,
                                                class_mode='binary')
        #Creating validation set
        valid_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'valid_image_paths.csv'),
                                                directory=dataset_root.parent,
                                                target_size=(224, 224),
                                                shuffle = False,
                                                batch_size=512,
                                                class_mode='binary')


        self.train_generator = train_gen
        self.val_generator = valid_gen

        return self.train_generator, self.val_generator


    def plot_image(self,i):
        # plot raw pixel data
        pyplot.imshow(self.train_images[i], cmap=pyplot.get_cmap('gray'))
        # show the figure
        return pyplot.show()

