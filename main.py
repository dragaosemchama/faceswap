# coding: utf-8
import cv2
import numpy
import time
from PIL import Image, ImageOps, ImageDraw, ImageEnhance

def detect(img, classifier):
    """Função de detecção de padrão

        Entrada:
                img: uma imagem
                classifier: o classificador representando qual padrão deve ser
                            identificado

        Saída: as coordenadas dos retângulo que contém o padrão indicado
               pelo classificador e a imagem

    """
    # Determina as dimensões mínimas do subconjunto da imagem onde o
    # padrão deverá ser detectado.
    min_rectangle = (60,60)

    rects = classifier.detectMultiScale(img, 1.2, 3, minSize=min_rectangle)

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    """Desenha um retângulo sobre cada face identificada.

        Entrada:
                rects: as coordenadas dos retângulos a serem desenhados.
                img: a imagem.

        Retorno:
                A imagem de entrada, devidamente marcada.

    """

    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img

def crop(img, startx, endx, starty, endy):
    """Recorta uma imagem.

        Args:
            img: uma imagem representada por um numpy array.
            startx: pixel inicial na horizontal.
            endx: pixel final na horizontal.
            starty: pixel inicial na vertical.
            endy: pixel final na vertical.

        Retorno:
            A subimagem contida no retângulo especificado.

    """
    return img[starty:endy, startx:endx]

def resize(img, width=None, height=None):
    """Redimensiona uma imagem.

        Args:
            img: uma imagem representada por um numpy array.
            width: nova largura.
            height: nova altura.

        Retorno:
            A imagem de entrada, nas dimensões especificadas.

    """

    # Obtem altura e largura iniciais
    (h, w) = img.shape[:2];

    # Se apenas um dos parâmetros é dado, calcula o segundo de forma a
    # manter o aspect ratio.
    if width > 0 and height == None:
        height = int(h*width/w);
    elif height > 0 and width == None:
        width = int(w*height/h);
    elif height == None and width == None:
        return img;

    # Escolhe o método de interpolação baseado no tipo de operação
    if width*height < w*h:
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA);
    else:
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR);


# Os dois métodos abaixo foram extraídos de um pequeno script escrito por
# robertskmiles. O primeiro método obtem a cor média de uma imagem.
# O segundo método ajusta o brilho de uma imagem de forma que ela assuma a cor
# média especificada pelo parâmetro col. Link a seguir:
# https://gist.github.com/robertskmiles/3228852#file-face_replace-py

def meancol(source):
	"""Find the mean colour of the given image"""
	onepix = source.copy()
	onepix.thumbnail((1,1),Image.ANTIALIAS)
	return onepix.getpixel((0,0))

def adjust(im, col, startcol=None):
	"""Adjust the image such that its mean colour is 'col'"""
	if startcol is None:
		startcol = meancol(im)
	rband, gband, bband = im.split()
	rbri, gbri, bbri =  ImageEnhance.Brightness(rband), \
                        ImageEnhance.Brightness(gband), \
                        ImageEnhance.Brightness(bband)
	rband = rbri.enhance((float(col[0]) / float(startcol[0])))
	gband = gbri.enhance((float(col[1]) / float(startcol[1])))
	bband = bbri.enhance((float(col[2]) / float(startcol[2])))
	im = Image.merge("RGB",(rband, gband, bband))
	return im


# Programa principal
if __name__ == '__main__':

    # Carregamento um arquivo treinado para reconhecimento de padrões.
    # O arquivo selecionado é treinado para detecção de faces.
    # Este arquivo pode ser encontrado junto com o download do OpenCV.
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    # Inicia uma captura de vídeo a partir do primeiro dispositivo de vídeo
    # encontrado no computador. Os frames do vídeo serão capturados
    # usando o objeto camera.
    camera = cv2.VideoCapture(0)

    # Loop principal
    # Aqui, os frames serão continuamente capturados e processados
    while 1:
        # Captura e realiza operações sobre a imagem antes da detecção
        # captura um frame
        (_,frame) = camera.read()
        # redimensiona o frame, se necessário
        #frame = cv2.resize(frame, (320, 240))

        # Realiza uma detecção pelo padrão indicado no classificador sobre o
        # frame; desenha um retângulo sobre cada padrão identificado (opcional)
        (rects, frame) = detect(frame, classifier)
        #frame = box(rects, frame)

        # A troca de faces só será feita quando duas faces forem detectadas
        if len(rects) == 2:
            # Recorta as duas faces detectadas e armazena em face1 e face2
            f1x1, f1y1, f1x2, f1y2 = rects[0]
            face1 = crop(frame, f1x1, f1x2, f1y1, f1y2)
            f2x1, f2y1, f2x2, f2y2 = rects[1]
            face2 = crop(frame, f2x1, f2x2, f2y1, f2y2)

            # Redimensiona cada uma das faces; face1 assume as dimensões
            # de face2 e vice-versa.
            (h1, w1) = face1.shape[:2]
            (h2, w2) = face2.shape[:2]
            face1 = resize(face1, h2, w2)
            face2 = resize(face2, h1, w1)

            # Operações como ajuste de brilho e aplicação de máscara alpha
            # serão feitas utilizando PIL (Python Image Library), então criamos
            # uma representação das imagens como um objeto PIL Image.
            frame_pil = Image.fromarray(frame)
            face1_pil = Image.fromarray(face1)
            face2_pil = Image.fromarray(face2)

            # Calcula a cor média de cada uma das faces, aplica essa cor na
            # outra imagem; face1 terá a cor média de face2 e vice-versa.
            face1_avgcolor = meancol(face1_pil)
            face2_avgcolor = meancol(face2_pil)
            face1_pil = adjust(face1_pil, face2_avgcolor)
            face2_pil = adjust(face2_pil, face1_avgcolor)

            # Gera uma máscara alpha para cada face, baseada no padrão de
            # máscara estabelecido na imagem mask.png.
            mask1 = Image.open('mask2.png').resize((w2,h2))
            mask2 = Image.open('mask2.png').resize((w1,h1))

            # Troca as faces, aplicando a respectiva máscara alpha.
            # A face1 é posta nas coordenadas da face2 e vice-versa.
            frame_pil.paste(face1_pil, (f2x1, f2y1, f2x2, f2y2), mask1)
            frame_pil.paste(face2_pil, (f1x1, f1y1, f1x2, f1y2), mask2)

            # Converte a imagem no formato PIL para o formato aceito pelo
            # OpenCV (numpy array)
            frame = numpy.array(frame_pil)

        # Mostra a imagem na janela do programa
        cv2.imshow('Face Swap', frame)

        # Captura qualquer tecla pressionada, armazena o código em key
        key = cv2.waitKey(1)
        # Se a tecla for ESC, fecha o programa
        if key == 27:
            break
        # Se a tecla for S, salva o frame atual como imagem PNG.
        elif key == ord('s'):
            cv2.imwrite(str(time.time()).replace('.', '')+".png", frame)
            print("Imagem salva...")

    # Fecha a janela do programa
    cv2.destroyAllWindows()
