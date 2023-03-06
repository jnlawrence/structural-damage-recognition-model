import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt


# To load a file use the format [/mnt/c/path/to/file]
data = np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_test.npy')
print(data.shape)

vgg = VGG16(include_top = False,
            weights = 'imagenet', 
            input_tensor = None, 
            input_shape = (224,224,3), #shape of npy file data
            pooling = None,
            classes = 1000,
            classifier_activation="softmax") 

# Do not retrain convolutional layers
for layer in vgg.layers:
    layer.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

# To show all the data in npy file as images [FOR DEMO PURPOSES ONLY]
img_array=np.load(r'/mnt/c/Users/John/Documents/Datasets/task7/task7_X_test.npy')
i = 0
while i < len(img_array):
   plt.imshow(img_array[i], cmap='gray')
   plt.show()
   i += 1