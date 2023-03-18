import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

# To load a file use the format [/mnt/c/path/to/file]
data = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_train.npy')
print(data.shape)

vgg16_model = VGG16(include_top = False,
            weights = 'imagenet', 
            input_tensor = None, 
            input_shape = (224,224,3), #shape of npy file data
            pooling = None,
            classes = 1000,
            classifier_activation="softmax") 

# Do not retrain convolutional layers
for layer in vgg16_model.layers:
    layer.trainable = False

input_shape = keras.Input(shape=(224, 224, 3))

# Add new fully connected layers
x = Flatten()(vgg16_model.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of classes in your dataset

# Create a new model with the fully connected layers added
model = Model(inputs=vgg16_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, #training data directory
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir, #validation data directory
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Save the trained model
model.save('vgg16_trained.h5')

# To show all the data in npy file as images [FOR DEMO PURPOSES ONLY]
img_array=np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_test.npy')
i = 0
while i < len(img_array):
   plt.imshow(img_array[i])
   plt.show()
   i += 1
   img = Image.fromarray(img_array[i], 'RGB')
   img.show()
   img.save('sample.png')
   i += 1

""" 
load the VGG16 model without the fully connected layers using the 
'VGG16 function from Keras, and freeze all the layers in the pre-trained model.
hen, we add new fully connected layers to the model, compile it with the adam optimizer and
'categorical_crossentropy loss function, and define data generators for training and validation 
using the ImageDataGenerator function from Keras. 
Finally, we train the model using the fit_generator function from Keras 
and save the trained model using the save function."""

# labels = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_y_test.npy')
# n_rows = data.shape[0]
# w = data.shape[1]
# d = data.shape[2]
# n_validation = int(n_rows * 0.1)
# n_train = n_rows - n_validation

# X_train = data[:n_train]
# y_train = labels[:n_train]
# X_val = data[-n_validation:]
# y_val = labels[-n_validation:]

# x_train_bw, x_val_bw = np.mean(X_train, axis=-1), np.mean(X_val, axis=-1)
# x_train_final = np.reshape(x_train_bw, (n_train, w*d))
# x_val_final = np.reshape(x_val_bw, (n_validation, w*d))

# data = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_train.npy')
# labels = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_y_train.npy')
# # Categories = os.listdir(labels)
# # print('Total Number of Categories: ', len(Categories))
# # print('CategoriesList: ', Categories)
# print(data.shape)
# print(labels.shape)

