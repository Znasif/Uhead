import random
import numpy as np
import cv2
import sys
import imgaug as ia
from imgaug import augmenters as iaa

def getBack(nums):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-25, 25),
            # shear=(-8, 8)
        )
    ], random_order=True)
    image_size = 1024
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    img = ~img
    points = []
    for i in range(nums):
        x = random.randint(0, image_size)
        y = random.randint(0, image_size)
        points.append((x, y))
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1,1,2))
    cv2.polylines(img, [points], True, 0)
    img = seq.augment_images(img.reshape((1, img.shape[0], img.shape[1])))
    return img

if __name__ == "__main__":
    nums = int(sys.argv[1])
    getBack(nums)
