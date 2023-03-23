import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# To load a file use the format [/mnt/c/path/to/file]
# data = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_train.npy')
x_train = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_X_train.npy")
y_train = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_y_train.npy")

x_test = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_X_test.npy")
y_test = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_y_test.npy")
print(x_train.shape)

#Hyperparameters
num_classes = 3
batch_size = 1
num_epochs = 10

# Load the ResNet50 model without the fully connected layer
base_model = ResNet50(include_top = False,
            weights = 'imagenet', 
            input_tensor = None, 
            input_shape = (224,224,3), #shape of npy file data
            pooling = None,
            classes = 1000,
            classifier_activation="softmax")

# Freeze the ResNet50 layers so that they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the ResNet50 layers so that they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1./255,  
                                   rotation_range=40, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   brightness_range=(0.2, 0.7), 
                                   shear_range=45.0, 
                                   zoom_range=60.0,
                                   horizontal_flip=True, 
                                   vertical_flip=True
                                   )

# Fit the ImageDataGenerator on the training data
datagen.fit(x_train)

# Create generators for the training and Testing data
train_generator = datagen.flow(x_train,
                               y_train, 
                               batch_size=batch_size,
                               shuffle=True)
                               
test_generator = datagen.flow(x_test, 
                              y_test, 
                              batch_size=batch_size,
                              shuffle=False)

# Use the generators to train the model
history = model.fit(train_generator,
          validation_data=test_generator,
          epochs=10,
          steps_per_epoch= len(train_generator) // batch_size,
          validation_steps=1)

# Save the trained model
# model.save('resnet_trained.h5')

# Plot the training and Testing accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Testing'], loc='upper left')
plt.show()

# Plot the training and Testing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Testing'], loc='upper left')
plt.show()
