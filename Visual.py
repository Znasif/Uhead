import numpy as np
import matplotlib.pyplot as plt
from Process import Process
import cv2
import os
import random
import json
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa


class Visual:
    """
    This class mainly deals with visualising the different
    processes that take place
    """
    def __init__(self):
        pass

    in_folder = "Maps/"
    out_folder = "Extracted/"
    image = None
    current = [0, 0]
    shift = False
    clicks = []
    track = None

    @staticmethod
    def image_open(title, flag=0):
        """
        Open image file

        :param title: Open image with title as name
        :param flag: 0: default to binary 1: color image
        :return: numpy array of image
        """
        img = cv2.imread(Visual.in_folder+title, flag)
        Visual.image = img
        return img

    @staticmethod
    def image_write(title, img):
        """
        Write image to file

        :param title: name of image file to write to
        :param img: numpy array to write
        :return: NIL
        """
        Visual.image = img
        cv2.imwrite(Visual.out_folder+title.split(".")[0]+".tif", img)

    @staticmethod
    def plot(title, image, a_map='gray'):
        """
        Show image in new window

        :param title: Title of image
        :param image: the numpy array to represent
        :param a_map: default to greyscale image
        :return: NIL
        """
        height, width = image.shape[:2]
        _, ax = plt.subplots(1, figsize=(height//100, width//100))
        ax.set_title(title)
        if len(image.shape) == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap=a_map)
        ax.axis('off')
        Visual.image = image
        plt.show()

    @staticmethod
    def show(title, image):
        """
        Show image in new window with shifting window

        :param title: Title of image
        :param image: the numpy array to represent
        :return: NIL
        """
        Visual.image = image
        a, b = 600, 1200
        if Visual.shift:
            Visual.current[1] += b
            if Visual.current[1] > image.shape[1]:
                Visual.current[1] = 0
                Visual.current[0] += a
                if Visual.current[0] > image.shape[0]:
                    Visual.current[0] = 0
        mx, my = min(image.shape[0], Visual.current[0]+a), min(image.shape[1], Visual.current[1]+b)
        show_image = image[Visual.current[0]:mx, Visual.current[1]:my]
        cv2.imshow(title, show_image)

    @staticmethod
    def get_nums(im_org, im_plot):
        """
        Subtract plot contour image from original to get
        image with only numbers

        :param im_org: original image
        :param im_plot: plot contour image
        :return: numbered image
        """
        im_nums = ~(im_org ^ im_plot)
        # Visual.plot('Contour', im_plot)
        # Visual.plot('Numbers and Contour', im_org)
        Visual.plot('Numbers', im_nums)
        Visual.image = im_nums
        return im_nums

    @staticmethod
    def get_overlay(im_org, im_plot):
        """
        Shows overlap of the two images in different colors
        :param im_org: image 1
        :param im_plot: image 2
        :return: NIL
        """
        p, q = np.shape(im_org)
        new = np.zeros((p, q, 3), dtype=np.uint8)
        new = ~new
        new[im_plot == 0] = (255, 255, 0)
        new[im_org < 80] = (255, 0, 255)
        Visual.image = new
        return new

    @staticmethod
    def init_dict(direc = "Extracted/ALL/"):
        Visual.track = [[j for j in os.listdir(direc+i+"/")] for i in os.listdir(direc)]

    @staticmethod
    def on_mouse(event, x, y, flags, params):
        """
        Select seed Points using Left Mouse Click
        :param event: Left Mouse Click
        :param x: Column
        :param y: Row
        :param flags: NIL
        :param params: NIL
        :return: stores (y, x) into clicks
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(Visual.image, (Visual.current[1] + x, Visual.current[0] + y), 1, 0, -1)
            Visual.clicks.append((Visual.current[0] + y, Visual.current[1] + x))

    @staticmethod
    def get_pixel(img, title='Collect Seed'):
        """
        Selection of seed point for 1. Edge linking 2. Region growing etc.
        :param img: Original image
        :param title: Window Name
        :return: NIL
        """
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, Visual.on_mouse, 0)
        Visual.current = [0, 0]
        Visual.shift = False
        while True:
            Visual.show(title, img)
            Visual.shift = False
            pressed_key = cv2.waitKey(20) & 0xFF
            # print(chr(pressed_key))
            if pressed_key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif 47 < pressed_key < 58:
                b, c, _ = Process.region_growing(img, [Visual.clicks[-1]])
                nu = pressed_key - 48
                print(nu)
                Visual.image_write("ALL/" + str(nu) + "/" + str(Visual.track[nu]), c)
                Visual.track[nu] += 1
                continue
            elif pressed_key == ord('a'):
                Visual.shift = True
            elif pressed_key == ord('c'):
                Visual.connect_line()

    @staticmethod
    def connect_line():
        """
        Connect points from Clicks with 1. Lines 2. Extrapolated Curves
        :return: NIL
        """
        ln = len(Visual.clicks)
        for i in range(0, ln, 2):
            y, x = Visual.clicks[i]
            y_, x_ = Visual.clicks[i]
            if i + 1 < ln:
                y_, x_ = Visual.clicks[i+1]
            Visual.image = cv2.line(Visual.image, (x, y), (x_, y_), 0, 2)

    @staticmethod
    def plot_fidelity(x, y, seed_cnt):
        """
        Plot fitness functions vs seed_cnt
        :param x: fitness_function_1
        :param y: fitness_function_2
        :param seed_cnt: number of provided seeds
        :return: NIL
        """

    @staticmethod
    def getBack(img, nums):
        image_size = img.shape[0]
        seq = iaa.Sequential([
            # iaa.Fliplr(0.5), # horizontal flips
            # iaa.Crop(percent=(0, 0.1)), # random crops
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
        points = []
        for i in range(nums):
            x = random.randint(0, image_size)
            y = random.randint(0, image_size)
            points.append((x, y))
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(img, [points], True, 0, random.randint(1, 2))
        img = seq.augment_images(img.reshape((1, img.shape[0], img.shape[1])))
        return img[0]

    @staticmethod
    def generate_data(nums, nums_per, subset, dimension):
        # random choice of digits from
        dir = "Extracted/ALL/"
        fls = len(Visual.track)
        # mask = [[] for i in range(fls)]
        annotations = {}
        for num in tqdm(range(nums)):
            a = np.zeros((dimension, dimension), dtype=np.uint8)
            a = ~a
            file_name = str(num) + ".tif"
            annotations[file_name] = {}
            annotations[file_name]["fileref"] = ""
            annotations[file_name]["size"] = dimension*dimension
            annotations[file_name]["filename"] = file_name
            annotations[file_name]["base64_img_data"] = ""
            annotations[file_name]["file_attributes"] = {}
            annotations[file_name]["regions"] = {}
            for i in range(nums_per):
                b = random.randint(0, fls - 1)
                ci = random.randint(0, len(Visual.track[b]) - 2)
                c = Visual.track[b][ci]
                e = dir + str(b) + "/" + c
                d = cv2.imread(e, 0)
                lx, ly = d.shape[:2]
                attempt = 50
                while attempt > 0:
                    x = random.randint(0, dimension - lx)
                    y = random.randint(0, dimension - ly)
                    if np.any(np.where(a[x:x+lx, y:y+ly] < 180)):
                        attempt -= 1
                    else:
                        a[x:x + lx, y:y + ly] = d
                        '''
                        ann_x, ann_y = [x, x+lx], [y, y+ly]
                        '''
                        approx = Process.get_contour(d, -1)
                        ann_x, ann_y = [], []
                        for poly in approx:
                            ann_x.append(int(y + poly[0][0]))
                            ann_y.append(int(x + poly[0][1]))
                        annotations[file_name]["regions"][i] = {}
                        annotations[file_name]["regions"][i]["shape_attributes"] = {}
                        annotations[file_name]["regions"][i]["region_attributes"] = b
                        annotations[file_name]["regions"][i]["shape_attributes"]["name"] = "polygon"
                        annotations[file_name]["regions"][i]["shape_attributes"]["all_points_x"] = ann_x
                        annotations[file_name]["regions"][i]["shape_attributes"]["all_points_y"] = ann_y
                        break
            a = Visual.getBack(a.copy(), 10)
            cv2.imwrite("numbers/data/" + subset + "/" + file_name, a)
        with open("numbers/data/" + subset + ".json", "w") as f:
            json.dump(annotations, f)
            print("Done")
