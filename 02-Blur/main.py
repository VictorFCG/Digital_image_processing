import sys
import numpy as np
import cv2
import os
import msvcrt

# ===============================================================================

INPUT_IMAGE_GS = "hand.bmp"
INPUT_IMAGE_CL = "flowers.bmp"

window = 7  # ingênua e integral

window_y = 7  # separável
window_x = 7


def media_ingenua(img, channels=1):  # channels = 3 para rgb
    w_range = int((window - 1) / 2)
    img_blurry = img.copy()
    win_area = window * window
    for z in range(channels):
        for y in range(w_range, img.shape[0] - w_range):
            for x in range(w_range, img.shape[1] - w_range):
                total = 0
                for y_win in range(y - w_range, 1 + y + w_range):
                    for x_win in range(x - w_range, 1 + x + w_range):
                        total += img[y_win, x_win, z]
                img_blurry[y, x, z] = total / win_area

    return img_blurry


def media_separavel(img, channels=1):
    y_range = int((window_y - 1) / 2)
    x_range = int((window_x - 1) / 2)
    img_blurry = img.copy()
    img_saida = img.copy()

    for z in range(channels):
        for y in range(img.shape[0]):
            fila = []
            total = 0
            for x in range(x_range, img.shape[1] - x_range):
                if x == x_range:
                    for x_win in range(x - x_range, 1 + x + x_range):
                        fila.append(img[y, x_win, z])
                        total += img[y, x_win, z]
                else:
                    total -= fila.pop(0)
                    total += img[y, x + x_range, z]
                    fila.append(img[y, x + x_range, z])
                img_blurry[y, x, z] = total / window_x
        for x in range(img.shape[1]):
            fila = []
            total = 0
            for y in range(y_range, img.shape[0] - y_range):
                if y == y_range:
                    for y_win in range(y - y_range, 1 + y + y_range):
                        fila.append(img_blurry[y_win, x, z])
                        total += img_blurry[y_win, x, z]
                else:
                    total -= fila.pop(0)
                    total += img_blurry[y + y_range, x, z]
                    fila.append(img_blurry[y + y_range, x, z])
                img_saida[y, x, z] = total / window_y

    return img_saida


def integral(img, channels=1):
    w_range = int((window - 1) / 2)
    img_int = img.copy()
    img_blurry = img.copy()
    win_area = window * window
    for z in range(channels):
        for y in range(1, img.shape[0]):
            img_int[y, 0, z] = img_int[y - 1, 0, z] + img[y - 1, 0, z]
        for x in range(1, img.shape[1]):
            img_int[0, x, z] = img_int[0, x - 1, z] + img[0, x - 1, z]
        for y in range(1, img.shape[0]):
            for x in range(1, img.shape[1]):
                img_int[y, x, z] = img_int[y - 1, x - 1, z] + img[y - 1, x - 1, z]

    for z in range(channels):
        for y in range(w_range, img.shape[0]):
            for x in range(w_range, img.shape[1]):
                img_blurry[y, x, z] = (
                    img_int[y, x, z]
                    + img_int[y - w_range, x - w_range, z]
                    - img_int[y, x - w_range, z]
                    - img_int[y - w_range, x, z]
                ) / win_area

    return img_blurry


def main():
    img = cv2.imread(INPUT_IMAGE_GS, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro abrindo a imagem 1.\n")
        sys.exit()
    img_color = cv2.imread(INPUT_IMAGE_CL)
    if img_color is None:
        print("Erro abrindo a imagem 2.\n")
        sys.exit()
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255
    img_color = img_color.astype(np.float32) / 255

    cv2.imshow("GS Original", img)
    cv2.waitKey()
    img_media = media_ingenua(img)
    cv2.imshow("GS ingenua", img_media)
    cv2.waitKey()
    img_sep = media_separavel(img)
    cv2.imshow("GS separavel", img_sep)
    cv2.waitKey()
    img_int = media_separavel(img)
    cv2.imshow("GS integral", img_int)
    cv2.waitKey()

    cv2.imshow("Cor", img_color)
    cv2.waitKey()
    img_color_media = media_ingenua(img_color, 3)
    cv2.imshow("Cor ingenua", img_color_media)
    cv2.waitKey()
    img_color_sep = media_separavel(img_color, 3)
    cv2.imshow("Cor separavel", img_color_sep)
    cv2.waitKey()
    img_color_int = media_separavel(img_color, 3)
    cv2.imshow("Cor integral", img_color_int)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
