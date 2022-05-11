from PIL import Image

import numpy as np
import os
dir = 'C:/bakalarka'
dir2 = 'pro'
dir3 = "pro2"
dir4 = 'C:/bakalarka/proresult_full'
x = 1
img = ""
background = ""

for img_name in os.listdir(dir + '/' + dir2):
    if "VG" in img_name:
        continue
    print(img_name)
    path_to_img = (dir + '/' + dir2 + '/' + img_name)
    img_name2 = img_name.replace("HE", "VG")
    if img_name2 not in os.listdir(dir + '/' + dir2):
        continue
    path_to_img_label = (dir + '/' + dir2 + '/' + img_name2)
    img = Image.open(path_to_img)
    background = Image.open(path_to_img_label)
    background = background.convert("RGBA")
    overlay = img.convert("RGBA")
    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(dir4 + '/' + img_name, "PNG")

