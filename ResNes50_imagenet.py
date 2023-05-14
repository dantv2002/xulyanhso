import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
# from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
plt.rcParams['figure.figsize']= (20,8)

batch_size = 32
img_size = 256

datagen = ImageDataGenerator(rescale=1/255.,
                             zoom_range =0.2,
                             validation_split = 0.15,
                             horizontal_flip = True,
                             rotation_range=20,
                             shear_range=0.2,
                            brightness_range = [0.6,1.2])

train_generator = datagen.flow_from_directory("/content/drive/MyDrive/grape_dataset/train",
                                              target_size= (img_size, img_size),
                                              batch_size = batch_size,
                                              subset = 'training',
                                              shuffle = True,
                                              class_mode ='categorical')

val_generator = datagen.flow_from_directory("/content/drive/MyDrive/grape_dataset/train",
                                              target_size= (img_size, img_size),
                                              batch_size = batch_size,
                                              subset = 'validation',
                                              shuffle = False,
                                              class_mode ='categorical')

test_generator = datagen.flow_from_directory("/content/drive/MyDrive/grape_dataset/test",
                                            target_size= (img_size, img_size),
                                            batch_size = batch_size,
                                            shuffle = False,
                                            class_mode ='categorical')

img_size = 256
base_model = ResNet50(include_top = False,
                      weights = 'imagenet',
                      input_shape = (img_size,img_size,3))

for layer in base_model.layers[:-8]:
    layer.trainable = False

last_output = base_model.output
x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.0001),  
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model_name = "grape_disease_model.h5"
checkpoint = ModelCheckpoint(model_name,
                            monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

try:
    history = model.fit(train_generator,
                           epochs=10,
                           validation_data=val_generator,
                           callbacks=[checkpoint,earlystopping])
except KeyboardInterrupt:
    print("\nTraining Stopped")

# my_model = tf.keras.models.load_model("/content/grape_disease_model.h5")

# y_test = test_generator.classes
# y_pred = my_model.predict(test_generator)
# y_pred = np.argmax(y_pred,axis=1)

