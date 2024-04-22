import numpy as np
import cv2

threshold = 0.5 #para o brightpass
sigma = 5
kernel = (2*sigma) + 1 #kernel
# ===============================================================================
def exibir(img1, img2, img3):
    cv2.imshow("Original", img1)
    cv2.imshow("Gauss", img2)
    cv2.imshow("Box", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()

# ===============================================================================
def bloom_gaus(hls):
    canal_0, canal_1, canal_2 = cv2.split(hls)
    #brightpass
    bpass = np.where(canal_1 < threshold, 0, canal_1)
    #blur
    total = bpass * 0
    for i in range(4):
        mult=pow(2, i)
        gblur = cv2.GaussianBlur(bpass, (0, 0), mult*sigma)
        total += gblur    
    #limitando aumento de intensidade e testando parâmetros
    total = np.where(total > 0.12, 0.12, total)
    canal_1 += total
    #truncamento
    canal_1 = np.where(canal_1 > 1, 1, canal_1)
    final = cv2.merge((canal_0, canal_1, canal_2))    
    final = cv2.cvtColor(final, cv2.COLOR_HLS2BGR)
    return final
# ===============================================================================
def bloom_box(hls):
    canal_0, canal_1, canal_2 = cv2.split(hls)
    #brightpass
    bpass = np.where(canal_1 < threshold, 0, canal_1)
    #blur
    total = bpass * 0
    for i in range(4):
        mult=pow(2, i)
        boxblur=bpass
        for j in range(5):
            boxblur = cv2.blur(boxblur, (1+(mult*kernel), 1+(mult*kernel)))
        total += boxblur   
    total = np.where(total > 0.12, 0.12, total)
    canal_1 += total
    canal_1 = np.where(canal_1 > 1, 1, canal_1)
    final = cv2.merge((canal_0, canal_1, canal_2))    
    final = cv2.cvtColor(final, cv2.COLOR_HLS2BGR)
    return final
            
# ===============================================================================
def main():
    img_gt2 = cv2.imread("GT2.BMP")
    img_wind = cv2.imread("Wind Waker GC.bmp")
    
    img_gt2 = img_gt2.astype(np.float32) / 255
    img_wind = img_wind.astype(np.float32) / 255
    
    hls_img_gt2 = cv2.cvtColor(img_gt2, cv2.COLOR_BGR2HLS)
    hls_img_wind = cv2.cvtColor(img_wind, cv2.COLOR_BGR2HLS)
    
    gauss_gt2 = bloom_gaus(hls_img_gt2)
    gauss_wind = bloom_gaus(hls_img_wind)
    
    box_gt2 = bloom_box(hls_img_gt2)
    box_wind = bloom_box(hls_img_wind)
    
    exibir(img_gt2, gauss_gt2, box_gt2)
    exibir(img_wind, gauss_wind, box_wind)

if __name__ == "__main__":
    main()

# ===============================================================================
