import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.measure import moments, moments_central, moments_normalized, moments_hu, centroid
import json

altura = 500
largura = 500


def processar_imagens_pasta(pasta):
    imagens_processadas = []
    imagens = []
    caminhos = []

    # Percorre todos os arquivos na pasta
    for nome_arquivo in os.listdir(pasta):
        # Verifica se o arquivo é uma imagem (extensões comuns: .jpg, .jpeg, .png)
        if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.jpeg') or nome_arquivo.endswith('.png') or nome_arquivo.endswith('.bmp'):
            caminho = os.path.join(pasta, nome_arquivo)
            caminhos.append(caminho)

    caminhos.sort()
    for caminho in caminhos:
        imagem = cv2.imread(caminho)

        if imagem is not None:
            # Pré-processamento da imagem
            # print(caminho)
            # input()
            leaflets, image_processed = segment_leaves(imagem)

            # Armazena a imagem processada e o caminho do arquivo
            imagens_processadas.append(leaflets)

            imagens.append(image_processed)

    return imagens_processadas, imagens, caminhos


def segment_leaves(imagem):
    # Conversão para tons de cinza
    imagem_processada = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Redimensionamento da imagem
    imagem_processada = cv2.resize(imagem_processada, (altura, largura))

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(imagem_processada, (5, 5), 0)

    # Otsu's thresholding
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate distance transform
    dist_transform = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)

    # Use a lower threshold for the foreground markers
    _, sure_fg = cv2.threshold(
        dist_transform, 0.2*dist_transform.max(), 255, 0)
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

    imagem = cv2.resize(imagem, (altura, largura))

    markers = cv2.watershed(imagem, markers)
    imagem[markers == -1] = [0, 0, 255]

    # Convert the markers to binary image
    binary = np.where((markers > 1), 255, 0).astype('uint8')

    # Apply a morphological erosion to create a gap between touching leaves
    kernel = np.ones((2, 2), np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Segment the leaves and store the images of individual leaves
    segmented_leaves = []
    contador = 0

    imagem_copy = imagem.copy()
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        leaf = imagem[y:y+h, x:x+w]
        segmented_leaves.append(leaf.copy())

        # # Calcular as coordenadas do centro do retângulo
        centro_x = int(x + w/2)
        centro_y = int(y + h/2)

        # Desenhar um retângulo ao redor do objeto
        cv2.rectangle(imagem_copy, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)

        # Adicionar texto com o contador no centro do retângulo
        texto = str(contador)
        tamanho_texto, _ = cv2.getTextSize(
            texto, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        texto_x = int(centro_x - tamanho_texto[0]/2)
        texto_y = int(centro_y + tamanho_texto[1]/2)
        cv2.putText(imagem_copy, texto, (texto_x, texto_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Incrementar o contador
        contador += 1

    return segmented_leaves, imagem_copy


def extract_features(leaf):

    # Convert the image to grayscale
    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

    # Calculate color features
    # Median color for each channel
    median_color = np.median(leaf, axis=(0, 1))

    # Calculate texture features
    glcm = graycomatrix(
        gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    # Calculate shape features
    m = moments(gray)
    cr, cc = centroid(m)  # center of mass coordinates
    m_central = moments_central(gray, [cr, cc])
    m_normalized = moments_normalized(m_central)
    hu_moments = moments_hu(m_normalized)

    # Calculate size features
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)

    # Package features into a dictionary
    features = {
        'median_color': median_color.tolist(),
        'dissimilarity': dissimilarity.tolist(),
        'homogeneity': homogeneity.tolist(),
        'energy': energy.tolist(),
        'correlation': correlation.tolist(),
        'hu_moments': hu_moments.tolist(),
        'area': area,
        'perimeter': perimeter,
    }

    return features


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
        # Extrai o nome do arquivo da imagem
        nome_arquivo = os.path.basename(caminhos[i])
        caminho_destino = os.path.join(pasta_destino, nome_arquivo)

        # Salva a imagem processada
        cv2.imwrite(caminho_destino, imagens[i])


def generate_variants(imagem):
    variants = []

    for i in range(0, 400, 100):
        if i == 0:
            resized = imagem.copy()
        else:
            resized = cv2.resize(imagem, (i, i), interpolation=cv2.INTER_AREA)
            variants.append(resized)
        rotated = resized.copy()
        images_rotated = []
        for i in range(0, 360, 90):
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
            variants.append(rotated)
            images_rotated.append(rotated)

        # Altera o brilho da imagem
        for rotated in images_rotated:
            brightness = rotated.copy()
            for i in range(0, 45, 10):
                brightness = cv2.convertScaleAbs(rotated, beta=i)
                variants.append(brightness)

        for rotated in images_rotated:
            brightness = rotated.copy()
            for i in range(0, 35, 10):
                brightness = cv2.convertScaleAbs(rotated, beta=-i)
                variants.append(brightness)

        for rotated in images_rotated:
            contrast = rotated.copy()
            for i in range(5, 10, 1):
                float_value = i/10
                contrast = cv2.convertScaleAbs(rotated, alpha=float_value)
                variants.append(contrast)

    return variants


def extrair_caracteristicas(imagens_processadas, caminhos):
    results = []
    names_leaflet = ["folhado", "araca",
                     "quaresmeira", "pessegueiro", "coleus", "uva"]

    for i, imagem in enumerate(imagens_processadas):
        results_local = []
        for j, contorno in enumerate(imagem):
            features = extract_features(contorno)
            results_local.append({
                "nome": '',
                "features": features
            })

        print("0 - Folhado | 1 - Araçá | 2 - Quaresmeira | 3 - Pessegueiro | 4 - Coleus | 5 - Uva")
        nome_arquivo = os.path.basename(caminhos[i])
        caminho_destino = os.path.join(pasta_destino, nome_arquivo)

        while True:
            values = input(f"{caminho_destino} ({len(imagem)}): ")

            values = values.split(" ")

            if len(values) == len(imagem):
                break
            # Verifica se todas as entradas são válidas
            else:
                for value in values:
                    if value not in ["0", "1", "2", "3", "4", "5"]:
                        print("Valor inválido!")
            print("Quantidade de valores inválida!")
        for j, value in enumerate(values):
            results_local[j]["nome"] = names_leaflet[int(value)]
        results.extend(results_local)

        for j, result in enumerate(results_local):
            variants = generate_variants(imagem[j])
            for variant in variants:
                features = extract_features(variant)
                results.append({
                    "nome": result["nome"],
                    "features": features
                })

    # max_len = 0
    # for result in results:
    #     results_in = result["features"]
    #     for key, value in results_in.items():
    #         # Verifica se é um array
    #         if type(value) == list and len(value) > max_len:
    #             max_len = len(value)

    # for i, result in enumerate(results):
    #     results_in = result["features"]
    #     for key, value in results_in.items():
    #         # Vai expandir os vetores para o mesmo tamanho
    #         if type(value) == list:
    #             results_in[key] = value + [0]*(max_len-len(value))
    #     results[i]["features"] = results_in

    # Salva os resultados em um TXT
    with open("./results.json", "w") as f:
        # Transforma em JSON
        f.write(json.dumps(results,
                           indent=4,
                           separators=(',', ': '),
                           ))


def test(imagens_processadas):
    images_test = [8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 7,
                   7, 7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

    for i, resultado in enumerate(imagens_processadas):
        res = len(resultado)
        if res != images_test[i]:
            print(
                f"Número de folhas na imagem {i+1}: {res} -> {images_test[i]}")


pasta_imagens = 'images'
pasta_destino = 'images_processed'

# Armazena imagens processadas e caminhos dos arquivos
imagens_processadas, imagens, caminhos = processar_imagens_pasta(
    pasta_imagens)

# Salva as imagens processadas na pasta de destino
salvar_imagens_processadas(imagens, caminhos, pasta_destino)

test(imagens_processadas)
# Extrai quantidade de folhas da imagem
resultados = extrair_caracteristicas(imagens_processadas, caminhos)
