from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import numpy as np

model = VGG16()

from keras.utils.vis_utils import plot_model
print(model.summary())
plot_model(model, to_file='vgg.png')
