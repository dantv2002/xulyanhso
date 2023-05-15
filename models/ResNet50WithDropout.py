import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ResNet50WithDropout:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = None
        self.test_generator = None
        self.model = None
        self.history = None

    def create_generators(self):
        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical'
        )
        self.test_generator = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical'
        )

    def build_model(self):
        # Define the input shape
        input_shape = (256, 256, 3)

        # Define the number of classes
        num_classes = 4

        # Define the model architecture
        inputs = tf.keras.Input(shape=input_shape)

        # Stage 1
        x = tf.keras.layers.Conv2D(64, (7,7), strides=(2,2), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

        # Stage 2
        shortcut = x
        x = tf.keras.layers.Conv2D(64, (1,1), strides=(1,1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), padding='valid')(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        # Stage 3
        shortcut = x
        x = tf.keras.layers.Conv2D(128, (1,1), strides=(2,2), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(512, (1,1), strides=(1,1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = tf.keras.layers.Conv2D(512, (1,1), strides=(2,2), padding='valid')(shortcut)
        if x.shape != shortcut.shape:
            shortcut = tf.keras.layers.Conv2D(512, (1,1), strides=(1,1), padding='valid')(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        #Stage 4
        shortcut = x
        x = tf.keras.layers.Conv2D(256, (1,1), strides=(2,2), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = tf.keras.layers.Conv2D(1024, (1,1), strides=(2,2), padding='valid')(shortcut)
        if x.shape != shortcut.shape:
            shortcut = tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='valid')(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        #Stage 5
        shortcut = x
        x = tf.keras.layers.Conv2D(512, (1,1), strides=(2,2), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(512, (3,3), strides=(1,1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(2048, (1,1), strides=(1,1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut = tf.keras.layers.Conv2D(2048, (1,1), strides=(2,2), padding='valid')(shortcut)
        if x.shape != shortcut.shape:
            shortcut = tf.keras.layers.Conv2D(2048, (1,1), strides=(1,1), padding='valid')(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        #Final layers
        x = tf.keras.layers.AveragePooling2D((7, 7))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(4, activation='softmax')(x)

        #Create Model
        self.model = tf.keras.models.Model(inputs=inputs, outputs=x)
    
    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
    def train_model(self, epochs=50):
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.test_generator
        )

    def evaluate_model(self):
        return self.model.evaluate(self.test_generator)

    def save_model(self, filename):
        self.model.save(filename)

def startTraining(train_dir, test_dir, epochs, filepath):
    custom_cnn = ResNet50WithDropout(train_dir, test_dir)
    custom_cnn.create_generators()
    custom_cnn.build_model()
    custom_cnn.compile_model()
    custom_cnn.train_model(epochs)
    evaluation = custom_cnn.evaluate_model()
    custom_cnn.save_model(filepath)
    return evaluation