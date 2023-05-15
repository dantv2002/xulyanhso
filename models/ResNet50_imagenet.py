import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

class ResNet50_imagenet:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = 32
        self.img_size = 256
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def create_generators(self):
        datagen = ImageDataGenerator(rescale=1/255.,
                                     zoom_range=0.2,
                                     validation_split=0.15,
                                     horizontal_flip=True,
                                     rotation_range=20,
                                     shear_range=0.2,
                                     brightness_range=[0.6,1.2])

        self.train_generator = datagen.flow_from_directory(self.train_dir,
                                                           target_size=(self.img_size, self.img_size),
                                                           batch_size=self.batch_size,
                                                           subset='training',
                                                           shuffle=True,
                                                           class_mode='categorical')

        self.val_generator = datagen.flow_from_directory(self.train_dir,
                                                         target_size=(self.img_size, self.img_size),
                                                         batch_size=self.batch_size,
                                                         subset='validation',
                                                         shuffle=False,
                                                         class_mode='categorical')

        self.test_generator = datagen.flow_from_directory(self.test_dir,
                                                          target_size=(self.img_size, self.img_size),
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          class_mode='categorical')

    def build_model(self):
        base_model = ResNet50(include_top=False,
                              weights='imagenet',
                              input_shape=(self.img_size, self.img_size, 3))

        for layer in base_model.layers[:-8]:
            layer.trainable = False

        last_output = base_model.output
        x = GlobalAveragePooling2D()(last_output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(4, activation='softmax')(x)
        self.model = Model(inputs=base_model.inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def evaluate_model(self):
        evaluation = self.model.evaluate(self.test_generator)
        loss = evaluation[0]
        accuracy = evaluation[1]
        return loss, accuracy

    def train(self, epochs=10):
        # model_name = "grape_disease_model.h5"
        # checkpoint = ModelCheckpoint(savepath,
        #                               monitor="val_loss",
        #                               mode="min",
        #                               save_best_only=True,
        #                               verbose=1)

        # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)

        # try:
        #     self.history = self.model.fit(self.train_generator,
        #                                   epochs=epochs,
        #                                   validation_data=self.val_generator,
        #                                   callbacks=[checkpoint, earlystopping])
        # except KeyboardInterrupt:
        #     print("\nTraining Stopped")
        self.model.fit(self.train_generator,
                            epochs=epochs,
                            validation_data=self.val_generator)
    
    def save_model(self, filepath):
        self.model.save(filepath)

def startTraining(train_dir, test_dir, epochs, filepath):
    resnet = ResNet50_imagenet(train_dir, test_dir)
    resnet.create_generators()
    resnet.build_model()
    resnet.train(epochs)
    resnet.save_model(filepath)
    evaluate = resnet.evaluate_model()
    return evaluate



