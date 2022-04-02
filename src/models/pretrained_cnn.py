
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import DenseNet169
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, GlobalMaxPooling2D, Dropout,MaxPooling2D,Conv2D, BatchNormalization,GlobalAveragePooling2D
from tensorflow import keras


class PRETRAINED_CNN():

    def __init__(self, num_categories):
        self.num_categories = num_categories

    def baseline_model(self):

        #Still not talking about our train/test data or any pre-processing.

        self.model = Sequential()

        # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        #note that this layer will be set below as NOT TRAINABLE, i.e., use it as is
        desnenet = DenseNet169(include_top=False, weights = 'imagenet', input_shape=(224, 224, 3) )
        desnenet.trainable = False

        self.model.add(desnenet)
        self.model.add(Flatten())
        self.model.add(Dense(1, activation = 'sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=['binary_accuracy'],
        )
        
        return self.model

    def alternate_model(self):
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3) )
        base_model.trainable = False
        last = base_model.output

        downsamp = MaxPooling2D(pool_size=1, strides=1, padding='Valid')(last)

        recg_net = Conv2D(224, kernel_size=(3,3), padding='same', activation='relu')(downsamp)
        recg_net = BatchNormalization()(recg_net)
        recg_net = Conv2D(32, (1,1), padding='same', activation='relu')(recg_net)
        recg_net = Flatten()(recg_net)
        recg_net = Dense(1, activation= 'sigmoid')(recg_net)

        self.model = Model(base_model.input, recg_net)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=['binary_accuracy'],
        )
        return self.model