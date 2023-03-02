import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def pick_color(event, x, y, flags, param):
    global mask
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        delta = 2

        lower_hsv_bound = (0, 0, 0)
        upper_hsv_bound = (int(pixel[0]) + delta, int(pixel[1]) + delta, int(pixel[2]) + delta)
        part1 = cv2.inRange(hsv, lower_hsv_bound, upper_hsv_bound)

        lower_hsv_bound = (int(pixel[0]) - delta, int(pixel[1]) - delta, int(pixel[2]) + delta)
        upper_hsv_bound = (180, 255, 255)
        part2 = cv2.inRange(hsv, lower_hsv_bound, upper_hsv_bound)

        temp = cv2.bitwise_or(part1, part2)
        mask = cv2.bitwise_or(temp, mask)
        plt.imshow(mask, cmap='gray')
        plt.show()
    if event == cv2.EVENT_RBUTTONDOWN:
        pixel = hsv[y, x]
        lower_hsv_bound = (int(pixel[0]) - 2, int(pixel[1]) - 2, int(pixel[2]) - 2)
        upper_hsv_bound = (180, 255, 255)
        temp = cv2.inRange(hsv, lower_hsv_bound, upper_hsv_bound)
        temp = cv2.bitwise_not(temp)

        mask = cv2.bitwise_or(temp, mask)
        plt.imshow(mask)
        plt.show()


def get_carotid_mask(img):
    global hsv, hues, sats, vals, mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:-1], np.uint8)

    cv2.namedWindow('BGR', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('BGR', 800, 800)
    cv2.imshow("BGR", img)
    cv2.setMouseCallback("BGR", pick_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imshow(mask)
    plt.show()

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # filter out small background elements
    min_size = 800
    # your answer image
    result_bg = np.zeros(mask.shape, np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            result_bg[output == i + 1] = 255

    result_bg = cv2.bitwise_not(result_bg)

    # filter out small foreground elements
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(result_bg, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    min_size = 800
    # your answer image
    result_fg = np.zeros(mask.shape, np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            result_fg[output == i + 1] = 255

    result_fg = cv2.medianBlur(result_fg, 3)
    result_fg[result_fg > 0] = 255

    plt.imshow(result_fg)
    plt.show()

    cv2.imwrite(masks_dir + "/" + im, result_fg)


if __name__ == '__main__':
    # tmava data
    # working_dir = "C:/projects/phd/svetla_tmava/tmava"
    # upper_hsv_bound = (235/2, (25/100) * 255, (90/100) * 255)
    # svetla data
    working_dir = "/data/ateroskleroza/svetla_tmava/tmava"
    imgs = os.listdir(working_dir + "/" + "imgs")
    masks_dir = working_dir + "/" + "plaque"
    for im in imgs:
        print("processing " + im)
        im_path = working_dir + "/" + "imgs/" + im
        if not os.path.exists(masks_dir + "/" + im):
            get_carotid_mask(cv2.imread(working_dir + "/" + "imgs/" + im))
        else:
            print("skipping " + im)
    print("run script done!")
    exit(0)


