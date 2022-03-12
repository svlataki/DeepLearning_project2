

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
from sklearn.utils import class_weight



class DataBuilder():

    def to_grayscale_then_rgb(image):
        image = tf.image.rgb_to_grayscale(image)
        return image

    def get_data(self):
        
        # load dataset
        
        dataset_root = Path(r'C:\Users\svlataki\Downloads\MURA-v1.1')
        #Dividing the image data generated into train set and validation set
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255, validation_split=0.25)
        
        train_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'train_image_paths.csv'),
                                                directory=dataset_root.parent,
                                                target_size=(224, 224),
                                                subset = 'training',
                                                shuffle = True,
                                                class_mode='binary')

        class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(train_gen.classes),y = train_gen.classes )
        class_weights_dict = dict(zip(np.unique(train_gen.classes), class_weights))
        
        #Creating validation set
        valid_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'train_image_paths.csv'),
                                                directory=dataset_root.parent,
                                                target_size=(224, 224),
                                                subset = 'validation',
                                                shuffle = True,
                                                class_mode='binary')

        test_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'valid_image_paths.csv'),
                                                directory=dataset_root.parent,
                                                target_size=(224, 224),
                                                shuffle = False)

        self.train_generator = train_gen
        self.val_generator = valid_gen
        self.test_generator = test_gen

        return self.train_generator, self.val_generator, self.test_generator,class_weights_dict


    def plot_image(self,i):
        # plot raw pixel data
        pyplot.imshow(self.train_images[i], cmap=pyplot.get_cmap('gray'))
        # show the figure
        return pyplot.show()

