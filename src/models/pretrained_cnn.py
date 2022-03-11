
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, GlobalMaxPooling2D, Dropout
from tensorflow import keras


class PRETRAINED_CNN():

    def __init__(self, in_shape, num_categories):
        self.in_shape = in_shape
        self.num_categories = num_categories

    def baseline_model(self):

        #Still not talking about our train/test data or any pre-processing.

        

        # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        #note that this layer will be set below as NOT TRAINABLE, i.e., use it as is
        desnenet = ResNet50(include_top=False, weights = 'imagenet', input_shape=(224, 224, 3) )
        desnenet.trainable = False

        x = desnenet.output
        x = GlobalMaxPooling2D()(x)
        x=Dropout(0.3)(x) 
        x=Dense(1024,activation='relu')(x) 
        x=Dropout(0.3)(x) 
        #x=Dense(1024,activation='relu')(x) 
        x=Dense(512,activation='relu')(x) 
        x=Dropout(0.3)(x) 
        x=Dense(1, activation= 'sigmoid')(x)
        self.model = Model(inputs = desnenet.input, outputs = x)


        # self.model.add(desnenet)
        # self.model.add(Flatten())
        # self.model.add(Dense(1, activation = 'sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            metrics=['binary_accuracy'],
        )
        # print(self.model.summary())
        return self.model
