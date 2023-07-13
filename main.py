import cv2
import os
import numpy as np

def preprocessamento(imagem, largura, altura):
    # Conversão para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Redimensionamento da imagem
    imagem_redimensionada = cv2.resize(imagem_cinza, (largura, altura))
    
    # Aplicar um filtro de suavização para reduzir ruídos
    imagem_suavizada = cv2.GaussianBlur(imagem_redimensionada, (5, 5), 0)

    # Normalização dos valores de pixel para o intervalo [0, 1]
    # imagem_normalizada = imagem_redimensionada.astype(np.float32) / 255.0

    return imagem_suavizada

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
                imagem_processada = preprocessamento(imagem, largura, altura)
                
                # Armazena a imagem processada e o caminho do arquivo
                imagens_processadas.append(imagem_processada)
                caminhos.append(caminho)

    return imagens_processadas, caminhos

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
        nome_arquivo = os.path.basename(caminhos[i])  # Obtém o nome do arquivo da imagem original
        caminho_destino = os.path.join(pasta_destino, nome_arquivo)

        cv2.imwrite(caminho_destino, imagens[i])
        # print(f"Imagem processada {i+1}/{len(imagens)} salva em '{caminho_destino}'.")


def test(resultados):
    images_test=[8,8,6,6,6,6,7,7,7,7,8,8,8,8,8,7,7,7,7,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8]

    count = 0
    for i, resultado in enumerate(resultados):
        count += images_test[i] - resultado
    
    print(count)
    


def extrair_caracteristicas(imagens_processadas):
    resultados = []
    
    for imagem in imagens_processadas:  
        # Detecção de bordas usando o operador de Canny
        bordas = cv2.Canny(imagem.astype(np.uint8), 50, 150)
        
        # Encontrar contornos na imagem
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos que se parecem com folhas
        folhas = []
        for contorno in contornos:
            # Calcular área do contorno
            area = cv2.contourArea(contorno)
            
            # Ajustar um limiar para considerar como folha
            if area > 30:  # Valor de limiar ajustável
                folhas.append(contorno)

        # Cálculo do número de folhas com base no contorno
        numero_folhas = len(folhas)
        
        resultados.append(numero_folhas)
    
    return resultados

pasta_imagens = 'images'
pasta_destino = 'images_processed'
largura_desejada = 254
altura_desejada = 254

# Armazena imagens processadas e caminhos dos arquivos
imagens_processadas, caminhos = processar_imagens_pasta(pasta_imagens, largura_desejada, altura_desejada)

# Salva as imagens processadas na pasta de destino
salvar_imagens_processadas(imagens_processadas, caminhos, pasta_destino)

# Extrai quantidade de folhas da imagem
resultados = extrair_caracteristicas(imagens_processadas)

test(resultados)

# for i, resultado in enumerate(resultados):
#     print(f"Número de folhas na imagem {i+1}: {resultado}")