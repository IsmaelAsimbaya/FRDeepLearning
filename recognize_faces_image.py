# importamos los paquetes necesarios
import face_recognition
import argparse
import pickle
import cv2

# construimos el analizador de argumentos
ap = argparse. ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="ruta de acceso a la base de datos serializada de "
                                                         "codificaciones faciales")
ap.add_argument("-i", "--image", required=True, help="ruta a la imagen de entrada")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="Modelo de detección de rostros para usar: "
                                                                          "'HOG' o 'CNN'")
args = vars(ap.parse_args())

# el modulo de face_recognition realizara el trabajo pesado de identificacion facial
# mientras que OpenCV nos ayudara a cragar, convertir y mostrar la iamgen procesada.

# se analizaran tres argumentos de linea de comando
#    --encodings: La ruta al archivo pickle que contiene nuestras codificaciones faciales.
#    --imagen: La imagen que está siendo sometida al reconocimiento facial.
#    --detection-method: se elige el metodo de deteccion, hog para velocidad y cnn para precision

# en caso de no tener una GPU en el equipo que ejecutara el modelo se recomienda utilizar el modo hog
# ya que la ejecucion de cnn tomara mucho tiempo y  memoria.

# cargamos las codificaiones precalculadas + nombres de cara, posteriormente construimoc la codificaion de
# cara 128d para la imagen de entrada.

# carga de caras conocidas e incrustaciones
print("[INFO] cargando codificaciones...")
data = pickle.loads(open(args["encodings"], "rb").read())

# cargamos la imagen de entrada y la convertimso de BRG a RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detectamos las coordenadas (x,y) de los cuadros delimitadores correspondientes
# a cada cara de la imagen de entrada, luego calculamos las incrustaciones faciales
# para cada cara
print("[INFO] reconociendo rostros...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# inicializamos la lista de nombres para cada cara detectada
names = []

# bucle para las incrustaciones faciles
for encoding in encodings:
    # comparamos cada cra en la imagen de entrada con las codificaciones conocidas
    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.54)
    name = "Unknown"

    # Reconocimiento Facial
    # se buscan las coincidencias en la imagen de entrada (encodings) a nuestro conjunto de datos de codificaiones
    # conocidas (almacenadas en data["encodings"]) utilizando face_recognition.compare_faces
    # el uso de esta funcion devuelve una lista de valores true/false, para cada imagen de nuestro conjunto de datos.
    # la extencion de la lista dependera de la cantidad de imagenes que se encuentren en el archivo pickle de encodings.

    # Funcionamiento
    # internamente compare_faces calcula la distancia euclidiana entre la codificacion proporcionada y el conjunto de
    # datos que generamos previamente.
    #      - si la distncia esta por debajo de l tolerancia entonces devolvera true(mientras menor sea la tolerancia,
    #        mas estricto ser ael sistema de reconocimiento facial, se recomienda modificar este valor solo si dos
    #        individuos tienen rasgos muy parecidos).
    #      - si la distancia esta por encima de la tolerancia devolvemos false.

    # la implementacion proporcionada por la libreria utiliza un modelo "mas sofisticado" k-NN para realizar la
    # clasificacion, la libreria se basa en un dataset de rostros con 3M de datos cargado en la libreria dlib que
    # provee de los modelos para unicamente el reconocimiento de rostros en imagenes. la ventaja de utilizar este
    # dataset es su alta presicion al detectar rostros "in the wild" o situaciones que estan fuera de lo optimas
    # para implementaciones de reconocimiento facial en entornos reales, por otra parte la implementacion de dlib
    # posee un optimizacion a travez de cMake y el uso del lenguaje CUDA y cuDNN herramientas proporcionadas por
    # INVIDIA para procesamiento en GPU, aguilizando asi el procesamiento de redes neuronales para deep learning
    # (cuDNN - "cuda Depp Neural Networking"), el rendimeinto de la implementacion con aceleracion por GPU
    # reduce en enorme medidad el tiempo de analisis pasando de horas a un par de minutos, esto es un aspecto a tomar
    # en cuenta ya que al presentar un modelo que continuamente este validando el reconocimento facial para
    # el registro de asistencia se necesita el menor tiempo de computo posible con la mejor precision posible.
    # Ademas se toma en cuenta la variedad de dispositovos en los que la aplicacion puede ser implementada en donde
    # no se puede asegurar que las camaras sean de alta gama, proporcinando imagenes que no se encuentren en las mejores
    # condiciones para el reconocimiento facial, tomando esto en cuenta la implementacion de un modelo que nos
    # asegura una alta precision en entornos reales con tiempos computacionales bajos es una buena opcion para
    # proporcionar las bases de un sistema de reconocimiento facial escalable que puede crecer sobre si mismo al mejorar
    # el modelo con las imagenes recolectadas en el reconocimiento facial.

    # ... continuando con el reconocimiento ....
    # nuestra lista de matches tendra que computar el numero de "votos" para cada nombre (el numero de valores true
    # asociados a cada nombre) se cuentan los votos y se selecciona el nombre de la persona con la mayor cantidad
    # de votos correspondiente.

    # comprobamos si encontramosuna coincidencia
    if True in matches:
        # encontramos la posicion de todas las coincidencias e inicializamos un diccionario
        # que contara el numero total de veces que cada cara encontro una coincidencia
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # iteramos sobre los indices que coincidieron y mantenemos el contador para cada iteracion
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # determinamos que cara se reconocio con el mayor numero de votos
        # en caso de existir un empate python selecciona la primera entrada del diccionario
        name = max(counts, key=counts.get)

    # actualizamos la lista de nombres
    names.append(name)

# iteramos sobre las caras reconocidas
for ((top, right, bottom, left), name) in zip(boxes, names):
    # dibujamos el nombre de la imagen predecida
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# mostramos la imagen de salida
cv2.imshow("Image", image)
cv2.waitKey(0)

# C:\HardDisk\Biblioteca\Workspaces\Python\FRDeepLearning\venv\Scripts\python.exe recognize_faces_image.py --encodings encodings.pickle	--image examples/test.png
