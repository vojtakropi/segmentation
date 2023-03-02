

import cv2
import os
import numpy as np

def main():
    imageDict = 'D:\\bakalarka\\to_black\\images'

    mean = 0
    var = 35
    sigma = var ** 0.5
    # Convert an image from BGR to grayscale mode
    for image in os.listdir(imageDict):
        if "VG" in image:
            img=cv2.imread(imageDict +"\\"+image)
            gaussian = np.random.normal(mean, sigma, (512, 512))  # np.zeros((224, 224), np.float32)

            noisy_image = np.zeros(img.shape, np.float32)

            ksize = (5, 5)

            # Using cv2.blur() method
            blurred = cv2.blur(img, ksize)
            if len(img.shape) == 2:
                noisy_image = blurred + gaussian
            else:
                noisy_image[:, :, 0] = blurred[:, :, 0] + gaussian
                noisy_image[:, :, 1] = blurred[:, :, 1] + gaussian
                noisy_image[:, :, 2] = blurred[:, :, 2] + gaussian

            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)


            # Displaying the image
            # cv2.imshow('blurred', blurred)
            #
            # cv2.imshow("img", img)
            # cv2.imshow("gaussian", gaussian)

            cv2.imwrite("D:\\bakalarka\\to_gaussian\\images\\" + image, noisy_image)



if __name__ == "__main__":
    main()
