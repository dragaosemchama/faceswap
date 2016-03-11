# coding: utf-8
import cv2
import numpy
import time
import color_transfer

from PIL import Image, ImageOps, ImageDraw

# Carregamento um arquivo de treinamento para reconhecimento de padrões
# específicos na imagem em um objeto classificador. O arquivo selecionado
# contem as informações necessárias para detecção de faces.
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Função de detecção de padrão
# Entrada: uma imagem
# Saída:   as coordenadas de um retângulo que contem o padrão indicado
#          pelo classificador e a imagem
def detect(img):
    # Determina as dimensões mínimas do subconjunto da imagem onde o
    # padrão deverá ser detectado. Valores pequenos aumentam a distância de
    # visão do robô detector, mas tornam o processamento demorado.
    min_rectangle = (100,100)

    rects = cascade.detectMultiScale(img, 1.2, 3, minSize=min_rectangle)

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

# Função para desenhar um retângulo sobre cada padrão identificado
# Entrada: as coordenadas do retângulo que contem o padrão identificado
#          e a imagem que deve ser marcada;
# Saída    a imagem devidamente marcada.
def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img

def crop(img, startx, endx, starty, endy):
    """Crop an image.

        Args:
            img: a cv2 image.
            startx: starting pixel in x direction.
            endx: ending pixel in x direction.
            starty: starting pixel in y direction.
            endy: ending pixel in y direction.

        Returns:
            The input cv2 image, cropped.

        Raises:

    """
    return img[starty:endy, startx:endx]

def resize(img, width=None, height=None):
    """Resize an image.

        Args:
            img: a cv2 image.
            width: new width.
            height: new height.

        Returns:
            The input cv2 image, resized to new width and height.

        Raises:

    """

    # Get initial height and width
    (h, w) = img.shape[:2];

    # If just one of the new size parameters is given, keep aspect ratio
    if width > 0 and height == None:
        height = int(h*width/w);
    elif height > 0 and width == None:
        width = int(w*height/h);
    elif height == None and width == None:
        return img;

    # Choose interpolation method based on type of operation, shrink or enlarge
    if width*height < w*h:
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA);
    else:
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR);

# Programa principal
if __name__ == '__main__':

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
        # redimensiona o frame
        #frame = cv2.resize(frame, (320, 240))
        # altera o sistema de cores da imagem para grayscale (preto e branco)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Realiza uma detecção pelo padrão indicado no classificador sobre
        # o frame e desenha um retângulo sobre cada padrão identificado
        (rects, frame) = detect(frame)
        #frame = box(rects, frame)

        face1 = None
        face2 = None

        if len(rects) == 2:
            # Get face crops
            f1x1, f1y1, f1x2, f1y2 = rects[0]
            face1 = crop(frame, f1x1, f1x2, f1y1, f1y2)
            f2x1, f2y1, f2x2, f2y2 = rects[1]
            face2 = crop(frame, f2x1, f2x2, f2y1, f2y2)

            # Resize each image to the size of the other
            (h1, w1) = face1.shape[:2]
            (h2, w2) = face2.shape[:2]
            face1 = resize(face1, h2, w2)
            face2 = resize(face2, h1, w1)

            # Adjust brightness of each face
            face1 = color_transfer.color_transfer(face2, face1)
            face2 = color_transfer.color_transfer(face1, face2)

            # Alpha masking is easier with PIL, so convert to PIL images
            frame_pil = Image.fromarray(frame)
            face1_pil = Image.fromarray(face1)
            face2_pil = Image.fromarray(face2)

            # Generate alpha mask from file
            mask1 = Image.open('mask.png').resize((w2,h2))
            mask2 = Image.open('mask.png').resize((w1,h1))

            # Swap faces using alpha masks
            frame_pil.paste(face1_pil, (f2x1, f2y1, f2x2, f2y2), mask1)
            frame_pil.paste(face2_pil, (f1x1, f1y1, f1x2, f1y2), mask2)

            # Convert to OpenCV image format
            frame = numpy.array(frame_pil)

            """ PREVIOUS ATTEMPT
            # Generate masks
            mask1 = numpy.zeros((h2, w2, 3), numpy.uint8)
            mask2 = numpy.zeros((h1, w1, 3), numpy.uint8)
            cv2.circle(mask1,(int(h2/2),int(w2/2)),int(min(h2,w2)/2.5),(255,255,255), int(-1))
            cv2.circle(mask2,(int(h1/2),int(w1/2)),int(min(h1,w1)/2.5),(255,255,255), int(-1))
            mask1 = cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)
            mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)
            _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
            _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

            # Swap faces
            frame_bg = cv2.bitwise_and(frame[f1y1:f1y2, f1x1:f1x2],
                    frame[f1y1:f1y2, f1x1:f1x2], mask = cv2.bitwise_not(mask2))
            face2_fg = cv2.bitwise_and(face2, face2, mask = mask2)
            frame[f1y1:f1y2, f1x1:f1x2] = cv2.add(frame_bg, face2_fg)

            frame_bg = cv2.bitwise_and(frame[f2y1:f2y2, f2x1:f2x2],
                    frame[f2y1:f2y2, f2x1:f2x2], mask = cv2.bitwise_not(mask1))
            face1_fg = cv2.bitwise_and(face1, face1, mask = mask1)
            frame[f2y1:f2y2, f2x1:f2x2] = cv2.add(frame_bg, face1_fg)
            """

        # Mostra a imagem na janela do programa
        cv2.imshow('Face Swap', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):
            print("Saving picture...")
            cv2.imwrite(str(time.time()).replace('.', '')+".png", frame)

    # Fecha a janela do programa
    cv2.destroyAllWindows()
