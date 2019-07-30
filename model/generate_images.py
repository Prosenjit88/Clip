from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image folder")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default=20,
	help="# of training samples to generate")
args = vars(ap.parse_args())

 
# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(
	rotation_range=80,
        shear_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

total = 0

# construct the actual Python generator

def gen_images(image):
        print("[INFO] generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
	save_prefix="image", save_format="jpg")
 
        # loop over examples from our image data augmentation generator
        for image in imageGen:
                # increment our counter
                total += 1
 
                # if we have reached the specified number of examples, break
                # from the loop
                if total == args["total"]:
                        break


def main():
        print("[INFO] loading example image...")
        files =  os.listdir(args["image"])
        for file in files:
                image = load_img(os.path.join(args["image"],file))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                gen_images(image)


if __name__ == '__main__':
    main()
