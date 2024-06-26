# ===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
# -------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2

# ===============================================================================

INPUT_IMAGE = "arroz.bmp"

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 3
LARGURA_MIN = 3
N_PIXELS_MIN = 9


# ===============================================================================


def binariza(img, threshold):
    """Binarização simples por limiarização.
    Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal,
    binariza cada canal independentemente.
    threshold: limiar.
    Valor de retorno: versão binarizada da img_in."""

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x, 0] > threshold:
                img[y, x, 0] = 1
            else:
                img[y, x, 0] = 0
    return img
    """
    return np.where(img > threshold, 1, 0).astype(np.float32)
    """
    depois de vários erros descobri que a função não mantém o tipo e adicionei o astype(np.float32)
    """


# -------------------------------------------------------------------------------
def flood_fill(img, y, x, comp):
    queue = [(y, x)]  # Inicializa uma fila com as coordenadas iniciais
    while queue:
        y, x = queue.pop(0)  # Retira o primeiro elemento da fila
        if img[y, x, 0] == 1:
            img[y, x, 0] = comp["label"]
            comp["n_pixels"] += 1
            # Atualiza coordenadas
            if y > comp["B"]:
                comp["B"] = y
            if x > comp["R"]:
                comp["R"] = x
            if x < comp["L"]:
                comp["L"] = x
            # Adiciona vizinhos à fila
            if (y - 1) >= 0:
                queue.append((y - 1, x))  # Cima
            if (y + 1) < img.shape[0]:
                queue.append((y + 1, x))  # Baixo
            if (x - 1) >= 0:
                queue.append((y, x - 1))  # Esquerda
            if (x + 1) < img.shape[1]:
                queue.append((y, x + 1))  # Direita
    


# -------------------------------------------------------------------------------
def rotula(img, largura_min, altura_min, n_pixels_min):
    """Rotulagem usando flood fill. Marca os objetos da imagem com os valores
    [0.1,0.2,etc].

    Parâmetros: img: imagem de entrada E saída.
                largura_min: descarta componentes com largura menor que esta.
                altura_min: descarta componentes com altura menor que esta.
                n_pixels_min: descarta componentes com menos pixels que isso.

    Valor de retorno: uma lista,
    onde cada item é um vetor associativo (dictionary)
    com os seguintes campos:

    'label': rótulo do componente.
    'n_pixels': número de pixels do componente.
    'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente
    conexo, respectivamente: topo, esquerda, baixo e direita."""

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    lista = []
    label = -1  # negativo para não ter preocupações com limite
    # não é usado em processamento, exibição etc
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x, 0] == 1:
                comp = {
                    "label": label,
                    "n_pixels": 0,
                    "T": y,
                    "L": img.shape[1],
                    "B": 0,
                    "R": 0,
                }
                label -= 1
                flood_fill(img, y, x, comp)
                # verificar
                if (
                    comp["n_pixels"] >= n_pixels_min
                    and (comp["B"] - comp["T"]) >= altura_min
                    and (comp["R"] - comp["L"]) >= largura_min
                ):
                    # append
                    lista.append(comp)
    return lista


# ===============================================================================


def main():
    # Abre a imagem em escala de cinza.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro abrindo a imagem.\n")
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem
    # ser colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza(img, THRESHOLD)
    cv2.imshow("01 - binarizada", img)
    cv2.imwrite("01 - binarizada.png", img * 255)

    start_time = timeit.default_timer()
    componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len(componentes)
    print("Tempo: %f" % (timeit.default_timer() - start_time))
    print("%d componentes detectados." % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

    cv2.imshow("02 - out", img_out)
    cv2.imwrite("02 - out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
