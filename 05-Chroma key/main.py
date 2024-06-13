import numpy as np
import cv2
import os.path


def mixBGFG(img, chromask, bg):
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - chromask)
        bg[:, :, c] = bg[:, :, c] * (chromask)

    img_final = img + bg

    return img_final


def maskTweak_v1(chromask, img):
    # binarizando entre objeto e fundo
    ch_int = (chromask * 255).astype(np.uint8)
    th, a = cv2.threshold(ch_int, 0, 1.0, cv2.THRESH_OTSU)
    
    h = int(img.shape[0]/16)
    w = int(img.shape[1]/16)
    if ( h % 2 == 0):
        h -= 1
    if ( w % 2 == 0):
        w -= 1   
    morph_k = np.ones((w,h), np.uint8)
    bin = np.where(chromask > (th / 255), 1.0, 0)
    
    #clareando regioes claras, escurecendo escuras
    blurry = cv2.GaussianBlur(chromask, (w, h), 0, 0)
    bg = np.clip(blurry - chromask, 0, 1) * bin
    chromask += bg
    fg = np.clip(-blurry + chromask, 0, 1) * (1 - bin)
    chromask -= fg
    
    chromask = cv2.GaussianBlur(chromask, (5, 5), 0, 0)
    
    chromask = cv2.normalize(chromask, None, 0, 1, cv2.NORM_MINMAX)
    
    return chromask

def maskTweak_v2(chromask, img):
    # binarizando entre objeto e fundo
    ch_int = (chromask * 255).astype(np.uint8)
    th, a = cv2.threshold(ch_int, 0, 1.0, cv2.THRESH_OTSU)
           
    #boost de contraste estoura o fundo como branco
    th /= 255 
    chromask **= 2
    chromask = np.clip(chromask / th**2, 0, 1)
    
    return chromask


def maskCreate(img):
    # definindo áreas verdes
    chromask = img[:, :, 1] - np.where(
        img[:, :, 0] > img[:, :, 2], img[:, :, 0], img[:, :, 2]
    )
    chromask = np.clip(chromask, 0, 1)
    return chromask


def main():
    bg = cv2.imread(os.path.join("img", "bg.jpg")).astype(np.float32) / 255
    bg2 = cv2.imread(os.path.join("img", "bg2.bmp")).astype(np.float32) / 255

    for i in range(9):
        img_path = os.path.join("img", str(i) + ".bmp")
        img = cv2.imread(img_path).astype(np.float32) / 255
        h = img.shape[0]
        w = img.shape[1]
        bg_stretch = cv2.resize(bg, (w, h))
        bg2_stretch = cv2.resize(bg2, (w, h))

        chromask = maskCreate(img)
        img[:, :, 1] -= chromask   
        
        #normalização direta, pode ser feita no background
        chromask = cv2.normalize(chromask, None, 0, 1, cv2.NORM_MINMAX)
        
        #diminuindo ruído com blur
        #chromask = maskTweak_v1(chromask, img)
        
        #esticando contraste
        #chromask = maskTweak_v2(chromask, img)

        v1 = mixBGFG(np.copy(img), chromask, bg_stretch)
        v2 = mixBGFG(img, chromask, bg2_stretch)

        cv2.imshow("Mascara", chromask)
        cv2.waitKey()
        cv2.imshow("Processada v1", v1)
        cv2.waitKey()
        cv2.imshow("Processada v2", v2)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
