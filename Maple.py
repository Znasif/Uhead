import cv2
import json
import random
import skimage.draw
from tqdm import tqdm

f = open("numbers/data/train.json", "r")
s = f.read()
annotations = json.loads(s)

dir = "numbers/data/train/"
n = 20
fls = 10
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(fls)]
for i in tqdm(range(n)):
    file_name = str(i) + ".tif"
    a = cv2.imread(dir + "Gen/" + file_name)
    for j in annotations[file_name]["regions"].values():
        num = j["region_attributes"]
        ann_x = j["shape_attributes"]["all_points_x"]
        ann_y = j["shape_attributes"]["all_points_y"]
        rr, cc = skimage.draw.polygon(ann_y, ann_x)
        a[rr, cc] = color[num]
    cv2.imwrite(dir + "Gen/" + str(i) + "a.tif", a)
