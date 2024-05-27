import numpy as np
import cv2

def mixBGFG(img, img_mask, bg_stretch):
    fg_img = np.copy(img)
    fg_img[:,:,1] -= (img_mask)
    max = np.max(img_mask)
    bg_stretch/=(max)
    for c in range(3): 
        fg_img[:, :, c] = fg_img[:, :, c] * (1-img_mask) 
        bg_stretch[:, :, c] = bg_stretch[:, :, c] * (img_mask)
        
    img_final = fg_img + bg_stretch
    img_final = np.where(img_final > 1, 1, img_final) 
    img_final = np.where(img_final < 0, 0, img_final) 
             
    cv2.imshow("teste 1", img_final)
    cv2.waitKey()

def maskCreate(img):
    #definindo áreas verdes
    img_maskgb = img[:,:,1] - img[:,:,0]
    img_maskgr = img[:,:,1] - img[:,:,2]
    img_mask = np.where(img_maskgb < img_maskgr, img_maskgb, img_maskgr)
    #max = np.max(img_mask) Não usado
    img_mask = np.where(img_mask < 0, 0, img_mask)#/(max)
    img_mask = cv2.medianBlur(img_mask, (3))
    cv2.imshow("teste 1", img_mask)
    cv2.waitKey()
    
    return img_mask

def main():
    bg = cv2.imread("img\\bg.jpg").astype(np.float32)/255
    bg2 = cv2.imread("img\\bg2.bmp").astype(np.float32)/255
    
    for i in range(9):
        img_path = "img\\"+str(i)+".bmp"
        img = cv2.imread(img_path).astype(np.float32)/255
        h = img.shape[0]
        w = img.shape[1]
        bg_stretch = cv2.resize(bg, (w, h))
        bg_stretch2 = cv2.resize(bg2, (w, h))
        
        img_mask = maskCreate(img)
        mixBGFG(img, img_mask, bg_stretch)
        mixBGFG(img, img_mask, bg_stretch2)
        cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()