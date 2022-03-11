
from src.preprocessing.build_dataset import DataBuilder
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random

class Trainer:
    def __init__(self):
        self.history = None
        
        db = DataBuilder()
        db.get_data()

        self.train_generator, self.val_generator = db.get_data()


    def train(self, model):
        self.history = model.fit_generator(self.train_generator,
                                epochs = 1)
                                #validation_data=self.val_generator)

                        
        model.save('pretrained_cnn.h5')
    
    def predict(self,model):
        y_pred_probabilities = model.predict_generator(self.val_generator)
        print(y_pred_probabilities)
        y_pred_classes = np.where(y_pred_probabilities < 0.5, 0, 1)

        return y_pred_classes

    def predict_at_random(self, predictions):
        random_index = random.randint(0, len(predictions))
        random_prediction = predictions[random_index]
        random_image  = self.X_test[random_index].reshape( (28,28))
        random_correct = self.y_test[random_index]

        # plot raw pixel data
        fig, ax = plt.subplots(1,1)
        if random_correct == random_prediction:
            result = 'correctly'
        else:
            result = 'falsely'

        target_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        plt.title('The following image is '+ result +' predicted to be a : '+target_names[random_prediction], fontsize=8)
        plt.imshow(random_image, cmap=plt.get_cmap('gray'))
        
        # show the figure
        plt.show()


    def eval(self,model):
        eval = model.evaluate(self.X_test,self.y_test)
        print('Test loss is {}, Test accuracy is {}'.format(eval[0],eval[1]))

    def confusion_matrix(self, predictions):
        target_names=self.val_generator.class_indices.keys()
        ConfusionMatrixDisplay.from_predictions(np.array(self.val_generator.classes).astype(float),predictions, display_labels=target_names, xticks_rotation='vertical', cmap = 'RdPu')
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
        target_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        print(classification_report(self.y_test, predictions, target_names=target_names))

