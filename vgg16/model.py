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
batch_size = 16
num_epochs = 2

# Do not retrain convolutional layers
for layer in vgg16_model.layers:
    layer.trainable = False

input_shape = keras.Input(shape=(224, 224, 3))

# Add new fully connected layers
x = Flatten()(vgg16_model.output)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of classes in your dataset

# Create a new model with the fully connected layers added
model = Model(inputs=vgg16_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
                            #  rotation_range=20,
                            #  width_shift_range=0.1,
                            #  height_shift_range=0.1,
                            #  shear_range=0.2,
                            #  zoom_range=0.2,
                            #  horizontal_flip=True
                             )

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
          epochs=num_epochs,
          steps_per_epoch= len(train_generator) // batch_size,
          validation_data=test_generator)

# Save the trained model
model.save('vgg16_trained.h5')

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

print(history.history['val_accuracy'])
#Find the x and y position of the highest test accuracy
#list all val_accuracy values
list = history.history['val_accuracy']

#find highest val_accuracy
ymax =  max(history.history['val_accuracy'])

#find index of highest val_accuracy
xpos = list.index(max(history.history['val_accuracy']))

# Annotation for max accuracy
plt.annotate('Max Accuracy @ {}%'.format(round(ymax*100,2)), xy=(xpos, ymax), xytext=(xpos, ymax+.05), ha = 'center', 
             arrowprops=dict(arrowstyle="->", facecolor='black'))
#Show plot             
plt.show()
# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

predictions = model.predict(test_generator)
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
labels = ["Non-collapse", "Partial Collapse", "Global Collapse"]
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

disp.plot(cmap=plt.cm.Oranges)
plt.show()

# import ray
# from ray import air, tune
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.integration.keras import TuneReportCallback

# def tune_mnist(num_training_iterations):
#     sched = AsyncHyperBandScheduler(
#         time_attr="training_iteration", max_t=400, grace_period=20
#     )

#     tuner = tune.Tuner(
#         tune.with_resources(train_generator, resources={"cpu": 2, "gpu": 0}),
#         tune_config=tune.TuneConfig(
#             metric="mean_accuracy",
#             mode="max",
#             scheduler=sched,
#             num_samples=len(train_generator),
#         ),
#         run_config=air.RunConfig(
#             name="exp",
#             stop={"mean_accuracy": 0.99, "training_iteration": num_training_iterations},
#         ),
#         param_space={
#             "threads": 2,
#             "lr": tune.uniform(0.001, 0.1),
#             "momentum": tune.uniform(0.1, 0.9),
#             "hidden": tune.randint(32, 512),
#         },
#     )
#     results = tuner.fit()

#     print("Best hyperparameters found were: ", results.get_best_result().config)

# import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--smoke-test", action="store_true", help="Finish quickly for testing"
#     )
#     args, _ = parser.parse_known_args()

#     if args.smoke_test:
#         ray.init(num_cpus=4)

#     tune_mnist(num_training_iterations=5 if args.smoke_test else 300)

# x

""" 
load the VGG16 model without the fully connected layers using the 
'VGG16 function from Keras, and freeze all the layers in the pre-trained model.
hen, we add new fully connected layers to the model, compile it with the adam optimizer and
'categorical_crossentropy loss function, and define data generators for training and validation 
using the ImageDataGenerator function from Keras. 
Finally, we train the model using the fit_generator function from Keras 
and save the trained model using the save function."""