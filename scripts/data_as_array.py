import numpy as np
import wand.image
img_array = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_X_train.npy")
# img_array = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_y_train.npy")

i = 0
while i < len(img_array):
    print(img_array[i],i)
    i += 1
    img_array[i].show()