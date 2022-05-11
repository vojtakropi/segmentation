from PIL import Image
from fpdf import FPDF
import os
import unicodedata
import cv2
import numpy as np
from PIL import ImageFile
pdf = FPDF()
w, h = 0, 0
dir = 'C:/bakalarka'
dir2 = 'pro'
dir3 = "pro2"
dir4 = 'C:/bakalarka/proresult_full'
x = 1
img = ""
background = ""


def unicode_normalize(s):
    return unicodedata.normalize('NFKD', s).encode('utf-8', 'ignore')
t=0

for img_name, img_name2 in zip(os.listdir(dir + '/' + dir2), os.listdir(dir + '/' + dir3)):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    path_to_img = (dir + '/' + dir2 + '/' + img_name)
    path_to_img2 = (dir + '/' + dir3 + '/' + img_name)
    print(img_name)
    print(img_name2)
    if x == 1:
        w = 595
        h = 842
        pdf = FPDF(unit="pt", format=[w, h])
        x += 1
    image = path_to_img
    image2 = path_to_img2
    if x == 2:
        pdf.add_page()
        pdf.image(image, 300, 80, 300, 300)
        pdf.image(image2, 0, 80, 300, 300)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0,0,img_name2)
    if x == 3:
        pdf.image(image, 300, 380, 300, 300)
        pdf.image(image2, 0, 380, 300, 300)
        pdf.set_font('Arial', 'B', 12)
    if x == 2:
        x = 3
    else:
        x = 2
    t+=1
pdf.output("C:/bakalarka/output.pdf", "F")
