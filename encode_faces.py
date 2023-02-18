# Proyecto de Mineria

# realizamos las importaciones de paquetes necesarios
from imutils import paths
import face_recognition
import argparse  # argumentos de linea de comandos para procesamiento en tiempo de ejecucion
import pickle
import cv2
import os

# construimos el analizador de argumentos y analizador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="ruta al directorio de entrada de caras + imagenes")
ap.add_argument("-e", "--encodings", required=True,
                help="ruta de acceso a la base de datos serializada de codificaciones faciales")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="modelo de deteccion de rostros a usar: 'hog' o 'cnn'")
args = vars(ap.parse_args())
# utilizamos argparse para analizar argumentos de linea de comandos, cuando se ejecute el programa python
# en la linea de comandos, podmeos proporcionar informacion adicional al script sin salor de la terminal.
#   --dataset: Es la ruta del dataset
#   --encodings: las codificaciones faciales se escriben en el archivo al que se apunta en este arguemnto
#   --detection-method: antes de poder codificar rostros de imagenes primero se deben detectar. Para ello
#                       tenemos 2 metodos: 'hog' y 'cnn', estos son  los unicos valores que aceptara el
#                       argumento.

# Tomamos las rutas a los archivos del conjunto de datos
# Rutas a las imagenes de entrada del conjunto de datos
print("[INFO] cuantificando caras...")
imagePaths = list(paths.list_images(args["dataset"]))

# Inicializamos las listas de codificaciones conocidas y nombres conocidos
knownEncodings = []
knownNames = []

# bucle sobre las rutas de las imagenes
for (i, imagePath) in enumerate(imagePaths):
    # extraemos el nombre de la persona de la ruta de la imagen
    print("[INFO] procesando imagen {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # cargamos la imagen de entrada y la convertimos de BGR (OpenCV ordering) a dlib ordering(RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # este bucle se repetira por los n archivos que tengamos en nuestro dataset
    # extraemos el nombre de la persona con imagePath
    # cargamos laimagen al pasar el imagepath a cv2.imread
    # OpenCV ordena canales de color BGR, pero dlib espera RGB.
    # face_recognition usa el modulo dlib, asi que debemos intercambiar los espacios de color
    # se nombra a la nueva imagen rgb.

    # localizamos las codificaciones de cara y calculo

    # detectamos las coordenadas (x, y) de los cuadros delimitadores
    # correspondientes a cada cara en la imagen de entrada
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # calculamos la codificacion facial para cada cara
    encodings = face_recognition.face_encodings(rgb, boxes)

    # bucle sobre las codificaciones
    for encoding in encodings:
        # agregamos cada codificacion y su nombre a los conjuntos pre-establecidos
        knownEncodings.append(encoding)
        knownNames.append(name)

    # en este punto el bucle detectara una cara (o varias y asumira que son la misma, asi que se debe tener cuidaddo
    # con las imagenes que se proporcionan aldataset)

    # el metodo proporcionado por face_recognition face_loctions, requiere de dos componentes:
    #      - rgb: nuestra imagen
    #      - modelo: cnn o hog (proporcionado por nuestro diccionario de argumentos)
    # el sefundo metodo en face_encodings elcual codifica la cara en un vector

# volcamos las codificaiones + nombre al disco
print("[INFO] serializando codificaciones...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
# creamos un diccionario con dos llaves "encodings" y "names"
# volcamos los nombres y encodings al disco para futuros llamados
