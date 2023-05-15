import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BasicCNN:
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
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

    def train_model(self, epochs=50):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.test_generator
        )

    def evaluate_model(self):
        return self.model.evaluate(self.test_generator)

    def save_model(self, filepath):
        self.model.save(filepath)
        
def train(train_dir, test_dir, epochs, savepath):
    classifier = BasicCNN(train_dir, test_dir)
    classifier.create_generators()
    classifier.build_model()
    classifier.train_model(epochs)
    evaluation_results = classifier.evaluate_model()
    classifier.save_model(savepath)
    return evaluation_results