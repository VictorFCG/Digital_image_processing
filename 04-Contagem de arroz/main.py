import numpy as np
import cv2

# ===============================================================================
def flood_fill(img, y, x, comp):
    '''if img[y, x] == 1:
        img[y, x] = comp["label"]
        comp["n_pixels"] += 1
        
        if (y - 1) >= 0:
            flood_fill(img, y - 1, x, comp)  # cima
        if (y + 1) < img.shape[0]:
            flood_fill(img, y + 1, x, comp)  # baixo
        if (x - 1) >= 0:
            flood_fill(img, y, x - 1, comp)  # esquerda
        if (x + 1) < img.shape[1]:
            flood_fill(img, y, x + 1, comp)  # direita'''
    stack = [(y, x)]
    while stack:
        y, x = stack.pop()
        if img[y, x] == 1:
            img[y, x] = comp["label"]
            comp["n_pixels"] += 1
            if (y - 1) >= 0:
                stack.append((y - 1, x))  # cima
            if (y + 1) < img.shape[0]:
                stack.append((y + 1, x))  # baixo
            if (x - 1) >= 0:
                stack.append((y, x - 1))  # esquerda
            if (x + 1) < img.shape[1]:
                stack.append((y, x + 1))  # direita
# -------------------------------------------------------------------------------
def blob_list(img):
    flood_img = img.copy()
    lista = []
    label = (-1)  
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if flood_img[y, x] == 1:
                blob = {
                    "label": label,
                    "n_pixels": 0,
                }
                flood_fill(flood_img, y, x, blob)
                lista.append(blob)
                label -= 1   
    return lista
# ===============================================================================
def prep(file_name):#separação de fundo e remoção de ruído
    gs_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    
    #filtro da média para limiarização adaptativa.
    #kernel proporcional ao tamanho da imagem
    k_win=int(gs_img.shape[0]/16)
    avg_img = cv2.blur(gs_img, (k_win, k_win), borderType=cv2.BORDER_REFLECT)
    
    #comparando
    threshold=0.1
    adapt_img = np.where( (gs_img - avg_img) > threshold, 1, 0.0)
    
    #erodindo e dilatando (abertura/fechamento)
    morph_k = np.ones((5,5), np.uint8)
    erode_img = cv2.erode(adapt_img, morph_k, iterations=1)
    dilate_img = cv2.dilate(erode_img, morph_k, iterations=1)
    
    return dilate_img
# ===============================================================================
def count(img):
    #encontrando componentes
    #assumindo que só tenha restado grãos ou ao menos poucos objetos inválidos
    comp_list_ordered = sorted(blob_list(img), key=lambda x: x["n_pixels"])
    comp_qt = len(comp_list_ordered)
    
    index = int(comp_qt*0.4)
    index_f = comp_qt-index
    area = (comp_list_ordered[index_f]["n_pixels"] + comp_list_ordered[index]["n_pixels"])/2

    counter=0
    blob_area=0
    for i in range(comp_qt):
        if comp_list_ordered[i]["n_pixels"] < area:
            counter += 1
        else:
            blob_area += 1*comp_list_ordered[i]["n_pixels"]
        
    
    estim = int(blob_area/(area))
    return str(counter + estim)
    
# ===============================================================================
def main():
    img_60 = prep("60.bmp")
    img_82 = prep("82.bmp")
    img_114 = prep("114.bmp")
    img_150 = prep("150.bmp")
    img_205 = prep("205.bmp")
    
    count_60 = count(img_60)
    count_82 = count(img_82)
    count_114 = count(img_114)
    count_150 = count(img_150)
    count_205 = count(img_205)
    
    cv2.imshow("Real = 60, Contagem = " + count_60, img_60)
    cv2.waitKey()
    cv2.imshow("Real = 82, Contagem = " + count_82, img_82)
    cv2.waitKey()
    cv2.imshow("Real = 114, Contagem = " + count_114, img_114)
    cv2.waitKey()
    cv2.imshow("Real = 150, Contagem = " + count_150, img_150)
    cv2.waitKey()
    cv2.imshow("Real = 205, Contagem = " + count_205, img_205)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()

# ===============================================================================
