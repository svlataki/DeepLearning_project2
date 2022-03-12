
from src.preprocessing.build_dataset import DataBuilder
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import keras_tuner as kt
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

class Trainer:
    def __init__(self):
        self.history = None
        
        db = DataBuilder()
        db.get_data()

        self.train_generator, self.val_generator, self.test_generator,self.weights = db.get_data()
        
        self.STEP_SIZE_TRAIN=self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID=self.val_generator.n//self.val_generator.batch_size
        self.STEP_SIZE_TEST=self.test_generator.n//self.test_generator.batch_size

    def train(self, model):
        self.history = model.fit_generator(self.train_generator,
                                steps_per_epoch=self.STEP_SIZE_TRAIN,
                                validation_steps=self.STEP_SIZE_VALID,
                                epochs = 2,
                                validation_data=self.val_generator,
                                class_weight =self.weights)

    
        # pred=model.predict_generator(self.test_generator,
        #             steps=self.STEP_SIZE_TEST,
        #             verbose=1)

        #pd.DataFrame(pred).to_csv('predictions.csv',index=False)

    def predict(self,model):
        self.test_generator.reset()

        y_pred_probabilities = model.predict_generator(self.test_generator)
        print(y_pred_probabilities)
        y_pred_classes = np.where(y_pred_probabilities < 0.5, 0, 1)

        return y_pred_classes

    def eval(self,model):
        eval = model.evaluate_generator(generator=self.val_generator,
                                            steps=self.STEP_SIZE_TEST)     
        print('Test loss is {}, Test accuracy is {}'.format(eval[0],eval[1]))

    def confusion_matrix(self, predictions):
        target_names=self.test_generator.class_indices.keys()
        ConfusionMatrixDisplay.from_predictions(np.array(self.test_generator.classes).astype(float),predictions, display_labels=target_names, xticks_rotation='vertical', cmap = 'RdPu')
        plt.show()

    def plot_metrics(self):
        print(self.history.history)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def class_report(self,predictions):
        target_names=self.test_generator.class_indices.keys()
        print(classification_report(np.array(self.test_generator.classes).astype(float), predictions, target_names=target_names))

