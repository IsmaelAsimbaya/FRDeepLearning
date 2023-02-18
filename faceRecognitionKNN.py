from sklearn import neighbors
import math
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# entrena un clasificadro k vecinos mas crecanos para reconocimiento facial
def train(train_dir, model_save_path=None, n_neighbors=None, km_algo='ball_tre', verbose=False):
    # train_dir: directorio que contiene un subdirectorio para cada persona conocida con su nombre
    # model_save_path: directorio para guardar el modelo en el disco
    # n_neighbors: la estructura de datos subyacente para admitir knn.default es ball_tree
    # verbose: verbosidad del entrenamiento

    X = []
    y = []

    # Iteramos para cada persona en el conjunto de entrenamiento
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Iteramos para cada imagen de entrenamiento para la persona actual
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # si no existen personas (o existen demaciadas) en la imagen de entrenamiento, saltamos la imagen
                if verbose:
                    print("La imagen {} no es valida para entrenamiento: {}".format(img_path,
                                                                                    "No se encontro una cara" if len(
                                                                                        face_bounding_boxes) < 1 else "Se encontro mas de una cara"))
            else:
                # cargamos la codificacion de la cara actual al conjunto de entrenamiento
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # determinamos cuantos vecinos usar para el clasificador KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Eligiendo n_neighbors automaticamnete:", n_neighbors)

    # Crearmos y entrenamos el clasificador KNN
    knn_clsf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=km_algo, weights='distance')
    knn_clsf.fit(X, y)

    # guardamos el clasificador KNN entrenado
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clsf, f)

    return knn_clsf


# reconoce una imagen dadda usando un clasificadro KNN entrenado
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    return None
