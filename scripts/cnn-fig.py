import tensorflow as tf
from tensorflow.keras import layers, models
import visualkeras

from PIL import ImageColor

color_theme = {
    'InputLayer': '#0A192F',      # deep navy
    'Conv2D': '#0077B6',          # strong azure
    'MaxPooling2D': '#00B4D8',    # aqua blue
    'Dropout': '#90E0EF',         # soft cyan
    'BatchNormalization': '#CAF0F8',  # icy blue
    'Dense': '#023E8A',           # bold cobalt
    'Flatten': '#03045E',         # dark indigo
    'OutputLayer': '#0077B6',     # match Conv2D for visual closure
}

use_augmentation = False

model = models.Sequential()
model.add(layers.Input(shape=(128, 128, 3)))

if use_augmentation:
    model.add(layers.RandomRotation(0.2))
    model.add(layers.RandomFlip('horizontal'))
    model.add(layers.RandomZoom(0.2))

#Convolution Block 1
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

#Convolution Block 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.35))

#Fully connected layers
model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.35))

model.add(layers.Dense(128, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(3, activation='softmax'))

cnn_fig = visualkeras.layered_view(model, color_map = color_theme, legend=True, show_dimension=True)