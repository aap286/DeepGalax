import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy


# Construct an ImageDataGenerator object:
DIRECTORY = "Covid19-dataset/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256, 256)
BATCH_SIZE = 32

training_data_generator = ImageDataGenerator(rescale=1.0/255,

                                             # Randomly increase or decrease the size of the image by up to 10%
                                             zoom_range=0.1,

                                             # Randomly rotate the image between -25,25 degrees
                                             rotation_range=25,

                                             # Shift the image along its width by up to +/- 5%
                                             width_shift_range=0.05,

                                             # Shift the image along its height by up to +/- 5%
                                             height_shift_range=0.05,

                                             )

validation_data_generator = ImageDataGenerator()

train_iterator = training_data_generator.flow_from_directory(
    DIRECTORY, class_mode='categorical', color_mode='grayscale', batch_size=BATCH_SIZE)  # , subset='training')
train_iterator.next()

valid_iterator = validation_data_generator()
