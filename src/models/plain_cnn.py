
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten,GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout,MaxPooling2D,Conv2D, BatchNormalization,Input
from tensorflow import keras


# from keras import Input


class CNN():
    def __init__(self, in_shape, num_categories):
        self.in_shape = in_shape
        self.num_categories = num_categories

    def baseline_model(self):

        input = Input(shape=(224, 224, 3),name="input")
        x1 = Conv2D(32,(3,3),activation='relu')(input)

        x1 = Conv2D(64,(3,3),activation='relu')(x1)
        x1 = Conv2D(64,(3,3),activation='relu')(x1)
        x1 = MaxPooling2D((3, 3), strides=(2, 2))(x1)

        x2 = Conv2D( 64, (3, 3),activation='relu')(x1)
        x2 = Conv2D( 128, (3, 3),activation='relu')(x2)
        x2 = Conv2D(128, (3, 3),activation='relu')(x2)
        x2 = MaxPooling2D((3, 3), strides=(2, 2))(x2)

        x3 = Conv2D(128, (3, 3),activation='relu')(x2)
        x3 = Conv2D(256, (3, 3),activation='relu')(x3)
        x3 = Conv2D(256, (3, 3),activation='relu')(x3)
        x3 = MaxPooling2D((3, 3), strides=(2, 2))(x3)

        x4 = GlobalAveragePooling2D()(x3)
        x4 = Dropout(0.3)(x4)
        predictions = Dense(1,activation='sigmoid',name="final")(x4) #: here, not applying l2 reg.
        self.model = Model(inputs= input, outputs=predictions)

        ###Compiling model
        self.model.compile( loss='binary_crossentropy',optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=['binary_accuracy'])
        # print(self.model.summary())
        return self.model
    