# sudo apt install imagemagick

# importing the module
from PIL import Image
import os

dir_path = '../testbed/'
# get all the image paths in sorted order
image_paths = [] # = os.listdir(self.dir_path)
for file in os.listdir(dir_path):
  if file.endswith(".jpg"):
    image_paths.append(file)
all_images = [image_path.split('.')[0] for image_path in image_paths]
all_images = sorted(all_images)
os.chdir(dir_path)
for img in all_images:
    # importing the image
    im = Image.open(img+".jpg")
    # converting to jpg
    rgb_im = im.convert("RGB")
    # exporting the image
    rgb_im.save(img+".jpeg")
    # delete jpg
    os.remove(img+".jpg")
