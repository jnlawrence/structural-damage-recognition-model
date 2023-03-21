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
# data = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_train.npy')
x_train = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_X_train.npy")
y_train = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_y_train.npy")

x_test = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_X_test.npy")
y_test = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_y_test.npy")
print(x_train.shape)

vgg16_model = VGG16(include_top = False,
            weights = 'imagenet', 
            input_tensor = None, 
            input_shape = (224,224,3), #shape of npy file data
            pooling = None,
            classes = 1000,
            classifier_activation="softmax") 
num_classes = 3
batch_size = 32
num_epochs = 10

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

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

# Fit the ImageDataGenerator on the training data
datagen.fit(x_train)

# Create generators for the training and validation data
train_generator = datagen.flow(x_train,
                               y_train, 
                               batch_size=batch_size,
                               shuffle=True
                               )
                               
test_generator = datagen.flow(x_test, 
                              y_test, 
                              batch_size=batch_size,
                              shuffle=False)

# Use the generators to train the model
history = model.fit(train_generator,
          epochs=10,
          steps_per_epoch=len(train_generator) // batch_size,
          validation_data=test_generator)

# # Define data generators for training and validation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)
# #fully connected layer that will iterate over dataset

# train_generator = train_datagen.flow(
#     data, #training data directory
#     # target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='categorical')

# validation_generator = test_datagen.flow(
#     y, #validation data directory
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='categorical')

# # Train the model
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=num_epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size)

# Save the trained model
model.save('vgg16_trained.h5')

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# To show all the data in npy file as images [FOR DEMO PURPOSES ONLY]
# img_array=np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_test.npy')
# i = 0
# while i < len(img_array):
#    plt.imshow(img_array[i])
#    plt.show()
#    i += 1
#    img = Image.fromarray(img_array[i], 'RGB')
#    img.show()
#    img.save('sample.png')
#    i += 1

""" 
load the VGG16 model without the fully connected layers using the 
'VGG16 function from Keras, and freeze all the layers in the pre-trained model.
hen, we add new fully connected layers to the model, compile it with the adam optimizer and
'categorical_crossentropy loss function, and define data generators for training and validation 
using the ImageDataGenerator function from Keras. 
Finally, we train the model using the fit_generator function from Keras 
and save the trained model using the save function."""