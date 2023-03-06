from wand.image import Image
import os

SourceFolder="/mnt/c/Windows/System32/repos/thesis_raw_data/Photo_Database"
TargetFolder="/mnt/c/Windows/System32/repos/thesis_raw_data/Photo_Database"

# Convert all files with HEIC extension to JPEG
for file in os.listdir(SourceFolder):
    SourceFile=SourceFolder + "/" + file
    TargetFile=TargetFolder + "/" + file.replace(".HEIC",".JPG")

    img=Image(filename=SourceFile)
    img.format='jpg'
    img.save(filename=TargetFile)
    img.close()


# # Remove all HEIC files
# for file in os.listdir(SourceFolder):
#     if file.endswith(".HEIC"):
#         os.remove(SourceFolder + "/" + file)