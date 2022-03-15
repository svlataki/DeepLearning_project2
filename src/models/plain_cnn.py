
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten,GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout,MaxPooling2D,Conv2D, BatchNormalization
from tensorflow import keras
from keras import Input


class CNN():
    def __init__(self, in_shape, num_categories):
        self.in_shape = in_shape
        self.num_categories = num_categories

    def baseline_model(self):
        input = Input(shape=(299,299,3))
        x = Conv2D(input,32,(3,3),activation='relu')
        print(x)
        x = Conv2D(x,64,(3,3),activation='relu')
        x = Conv2D(x,64,(3,3),activation='relu')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = Conv2D(x, 64, (3, 3),activation='relu')
        x = Conv2D(x, 128, (3, 3),activation='relu')
        x = Conv2D(x, 128, (3, 3),activation='relu')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = Conv2D(x, 128, (3, 3),activation='relu')
        x = Conv2D(x, 256, (3, 3),activation='relu')
        x = Conv2D(x, 256, (3, 3),activation='relu')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1,activation='sigmoid',name="final")(x) #NOTE: here, not applying l2 reg.
        self.model = Model(inputs= input, outputs=predictions)

        ###Compiling model
        self.model.compile( loss='binary_crossentropy',optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy'])
        # print(self.model.summary())
        return self.model
    