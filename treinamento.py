import cv2
import os
import numpy as np
#Criação dos classificadores
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()
#Percorre a pasta "fotos" e retorna os respectivos IDs e imagens de cada pessoa
def getImagemComID():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #Listas para guardar faces e IDs
    faces = []
    ids = []
    #Converte as fotos lidas em escala de cinza e captura os IDs das fotos jogando dentro de uma lista
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)

    return np.array(ids), faces

ids, faces = getImagemComID()

print("Treinando...")
#Lê todas as imagens e executa o aprendizado
eigenface.train(faces, ids)
#Gera o arquivo e grava o treinamento no arquivo gerado pelo método Eigenface
eigenface.write('classificadorEigen.yml')
fisherface.train(faces, ids)
#Gera o arquivo e grava o treinamento no arquivo gerado pelo método Fisherface
fisherface.write('classificadorFisher.yml')
lbph.train(faces, ids)
#Gera o arquivo e grava o treinamento no arquivo gerado pelo método LBPH
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado")


