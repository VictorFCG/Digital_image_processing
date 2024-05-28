import numpy as np
import cv2
import os.path


def mixBGFG(img, img_mask, bg_stretch):
    fg_img = np.copy(img)
    fg_img[:, :, 1] -= img_mask
    max = np.max(img_mask)
    ajustada = img_mask / max

    for c in range(3):
        fg_img[:, :, c] = fg_img[:, :, c] * (1 - ajustada)
        bg_stretch[:, :, c] = bg_stretch[:, :, c] * (ajustada)

    img_final = fg_img + bg_stretch

    cv2.imshow("Mascara ajustada", (1 - ajustada))
    cv2.waitKey()
    cv2.imshow("Final", img_final)
    cv2.waitKey()
    cv2.destroyAllWindows()


def maskCreate(img):
    # definindo áreas verdes
    img_maskgb = img[:, :, 1] - img[:, :, 0]
    img_maskgr = img[:, :, 1] - img[:, :, 2]
    img_mask = np.where(img_maskgb < img_maskgr, img_maskgb, img_maskgr)
    # max = np.max(img_mask) Não usado
    img_mask = np.where(img_mask < 0, 0, img_mask)  # /(max)
    # Não notei melhoria significativa com blur, testei também com mediana e outros tamanhos de janela
    # img_mask = cv2.blur(img_mask, (3, 3))

    return img_mask


def main():
    bg = cv2.imread(os.path.join('img','bg.jpg')).astype(np.float32) / 255
    bg2 = cv2.imread(os.path.join('img','bg2.bmp')).astype(np.float32) / 255

    for i in range(9):
        img_path = os.path.join("img", str(i) + ".bmp")
        img = cv2.imread(img_path).astype(np.float32) / 255
        h = img.shape[0]
        w = img.shape[1]
        bg_stretch = cv2.resize(bg, (w, h))
        bg_stretch2 = cv2.resize(bg2, (w, h))

        img_mask = maskCreate(img)
        mixBGFG(img, img_mask, bg_stretch)
        mixBGFG(img, img_mask, bg_stretch2)


if __name__ == "__main__":
    main()
