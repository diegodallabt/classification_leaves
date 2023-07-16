import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from freeman_chain import extract_features


def preprocessamento(imagem, largura, altura):
    # Conversão para tons de cinza
    imagem_processada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Redimensionamento da imagem
    imagem_processada = cv2.resize(imagem_processada, (largura, altura))

    # Aplicar um filtro de suavização para reduzir ruídos
    imagem_processada = cv2.GaussianBlur(imagem_processada, (5, 5), 0)

    # Aplicar erosão nas imagens para remover ruídos
    imagem_processada = cv2.erode(imagem_processada, (3, 3), iterations=2)

    # Aplicar abertura e fechamento para remover ruídos
    # kernel = np.ones((3, 3), np.uint8)
    # imagem_processada = cv2.morphologyEx(imagem_processada, cv2.MORPH_OPEN, kernel)
    # imagem_processada = cv2.morphologyEx(imagem_processada, cv2.MORPH_CLOSE, kernel)

    # Normalização dos valores de pixel para o intervalo [0, 1]
    # imagem_normalizada = imagem_redimensionada.astype(np.float32) / 255.0

    return imagem_processada


def processar_imagens_pasta(pasta, largura, altura):
    imagens_processadas = []
    caminhos = []

    # Percorre todos os arquivos na pasta
    for nome_arquivo in os.listdir(pasta):
        # Verifica se o arquivo é uma imagem (extensões comuns: .jpg, .jpeg, .png)
        if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.jpeg') or nome_arquivo.endswith('.png') or nome_arquivo.endswith('.bmp'):
            caminho = os.path.join(pasta, nome_arquivo)
            imagem = cv2.imread(caminho)

            if imagem is not None:
                # Pré-processamento da imagem
                imagem_processada = segment_leaves(imagem)

                # Armazena a imagem processada e o caminho do arquivo
                imagens_processadas.append(imagem_processada)
                caminhos.append(caminho)

    return imagens_processadas, caminhos


def segment_leaves(imagem, min_contour_area=500):
    # Conversão para tons de cinza
    imagem_processada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Redimensionamento da imagem
    imagem_processada = cv2.resize(imagem_processada, (254, 254))

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(imagem_processada, (5, 5), 0)

    # Otsu's thresholding
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate distance transform
    dist_transform = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)

    # Use a lower threshold for the foreground markers
    _, sure_fg = cv2.threshold(
        dist_transform, 0.3*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Perform a dilation to improve the markers
    kernel = np.ones((2, 2), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=1)

    # Define the unknown region as the areas that are not clearly part of the background or the foreground
    _, sure_bg = cv2.threshold(
        dist_transform, 1.5*dist_transform.min(), 255, 0)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers of sure background
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    imagem = cv2.resize(imagem, (254, 254))

    markers = cv2.watershed(imagem, markers)
    imagem[markers == -1] = [0, 0, 255]

    # Convert the markers to binary image
    binary = np.where((markers > 1), 255, 0).astype('uint8')

    # Apply a morphological erosion to create a gap between touching leaves
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=10)

    # Apply a morphological dilation to restore the leaves to their original size
    dilated = cv2.dilate(eroded, kernel, iterations=10)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Segment the leaves and store the images of individual leaves
    segmented_leaves = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        leaf = imagem[y:y+h, x:x+w]
        segmented_leaves.append(leaf)

    return segmented_leaves


def criar_pasta(nome_pasta):
    if not os.path.exists(nome_pasta):
        os.makedirs(nome_pasta)
        print(f"A pasta '{nome_pasta}' foi criada com sucesso.")
    else:
        print(f"A pasta '{nome_pasta}' já existe.")


def salvar_imagens_processadas(imagens, caminhos, pasta_destino):
    # Cria a pasta de destino se ela não existir
    criar_pasta(pasta_destino)

    for i in range(len(imagens)):
        for image in imagens[i]:
            cv2.imshow('image', image)
            cv2.waitKey(0)

        # Obtém o nome do arquivo da imagem original
        # nome_arquivo = os.path.basename(caminhos[i])
        # caminho_destino = os.path.join(pasta_destino, nome_arquivo)

        # cv2.imwrite(caminho_destino, imagens[i])
        # print(f"Imagem processada {i+1}/{len(imagens)} salva em '{caminho_destino}'.")


def test(resultados):
    images_test = [8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                   7, 7, 7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

    count = 0
    for i, resultado in enumerate(resultados):
        # Calcular o módulo da diferença entre o resultado e o valor esperado
        count += abs(images_test[i] - resultado)

    print(count)


def extrair_caracteristicas(imagens_processadas):
    resultados = []
    extract_features(imagens_processadas)

    # for imagem in imagens_processadas:
    #     result = extract_features(imagem)
    #     print(result)
    #     input()


pasta_imagens = 'images'
pasta_destino = 'images_processed'
largura_desejada = 254
altura_desejada = 254

# Armazena imagens processadas e caminhos dos arquivos
imagens_processadas, caminhos = processar_imagens_pasta(
    pasta_imagens, largura_desejada, altura_desejada)

# Salva as imagens processadas na pasta de destino
# salvar_imagens_processadas(imagens_processadas, caminhos, pasta_destino)

# Extrai quantidade de folhas da imagem
resultados = extrair_caracteristicas(imagens_processadas)

# test(resultados)

images_test = [8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 7,
               7, 7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

# for i, resultado in enumerate(resultados):
#     print(f"Número de folhas na imagem {i+1}: {resultado} -> {images_test[i]}")
