from glob import glob
import os
import shutil

# Path setup.
path = r"C:\Users\John\Documents\Datasets\Consolidated_Photo_Database"
dest = r"C:\Users\John\Documents\Datasets\Photo_Database"

# Collect files.
files = glob(os.path.join(path, '**'), recursive=True)

# Ensure destination path exists.
if not os.path.isdir(dest):
    os.makedirs(dest)
# Move all files.
for f in files:
    if os.path.isfile(f):
        base = os.path.basename(f)
        shutil.move(f, os.path.join(dest, base))