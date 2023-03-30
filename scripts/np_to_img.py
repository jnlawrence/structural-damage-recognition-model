import numpy as np
from PIL import Image

data = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_X_train.npy")
y = np.load(r"/mnt/c/Windows/System32/repos/thesis_raw_data/task5/task5_y_train.npy")
# Confirm that they have the same number of data
print(data.shape, y.shape)

# print(data[0].shape) #4130 for 7, 1220 f0r 5
import PIL.ImageOps    
for i in range(len(data)-1220):
    # Floats within item array[i] are scaled from 0 to 255 to normalize the colors
    # as only changing the dtype to uint8 does not work because values goes wrong when 255 is exceeded 
    scaled = 255 * (data[i] - data[i].min())/(data[i].max() - data[i].min())
    im = Image.fromarray(scaled.astype(np.uint8))
 
    fname = '%05d_p%s.jpg' % (i, y[i])
    path = "/mnt/c/Windows/System32/repos/thesis_raw_data/task5/"
    im.save(f"{path}{fname}")