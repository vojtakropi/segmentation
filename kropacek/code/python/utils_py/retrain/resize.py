import cv2
import os


def main():
    imageDict = 'D:\\bakalarka\\to_black\\images_ultr'

    # Convert an image from BGR to grayscale mode
    for image in os.listdir(imageDict):
        img=cv2.imread(imageDict +"\\"+image)
        cv2.imwrite("D:\\bakalarka\\to_black\\ultr_resized\\"+image.replace(".png", ".tiff"), img)

if __name__ == "__main__":
    main()