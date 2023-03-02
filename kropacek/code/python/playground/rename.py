import os
from PIL import Image
import cv2
def main():
    directory  = "D:\\bakalarka\\all_unlabeled"
    for img in os.listdir(directory):
        if "HE" in img:
            originalImage = cv2.imread(directory + "\\" + img)
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            invert = cv2.bitwise_not(blackAndWhiteImage)
            cv2.imwrite("D:\\bakalarka\\to_black\\images2\\" + img, invert)

if __name__ == "__main__":
    main()
    exit(0)