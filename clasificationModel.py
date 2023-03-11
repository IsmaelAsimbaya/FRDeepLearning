from sklearn import neighbors
import math
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def trainKNN(X, y, model_save_path=None, n_neighbors=None, km_algo='ball_tree', verbose=False):

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Eligiendo n_neighbors automaticamnete:", n_neighbors)

    knn_clsf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=km_algo, weights='distance')
    knn_clsf.fit(X, y)

    if model_save_path is not None:
        print('guardando ...')
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clsf, f)

    return knn_clsf


def predict(X_img_path, knn_clsf=None, model_path=None, distance_threshold=0.54):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Direccion de imagen no valida: {}".format(X_img_path))

    if knn_clsf is None and model_path is None:
        raise Exception("Debe proporcionar el clasificador knn a través de knn_clf o model_path")

    # Cargamos el modelo (si se cargo uno)
    if knn_clsf is None:
        with open(model_path, 'rb') as f:
            knn_clsf = pickle.load(f)

    # cargamos la imagen y encontramos la posicion de las caras
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # si no encontramos caras en la imagen, retornamos una lista vacia
    if len(X_face_locations) == 0:
        return []

    # encontramos las codificaciones para las caras en la imagen de test
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # usamos el modelo KNN para encontrar las mejores coincidencias para la iamgen de test
    closest_distances = knn_clsf.kneighbors(faces_encodings, n_neighbors=1)
    print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(are_matches)

    # predecimos las clases y removemos las clasificaiones que no estan en el humbral
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clsf.predict(faces_encodings), X_face_locations, are_matches)]


if __name__ == "__main__":
    train_dir = "knn_examples/train"
    verbose = False

    X = []
    y = []

    print('transformando datos de entrada ...')
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

    print('... datos de entrada transformados')

    print(X[0])
    print(len(X[0]))
    print(type(X[0]))
    print(y)
    print(y[0])
    print(len(y[0]))
    print(type(y[0]))

    # Convertir X y y en un DataFrame de Pandas
    df = pd.DataFrame(data=X)
    df['etiquetas'] = y  # agregar las etiquetas como una columna adicional

    # Guardar el DataFrame como un archivo CSV
    df.to_csv('datos.csv', index=False)

    print('particionando datos ...')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('... datos particionados')
    print("X_Train: {}, Y_Train: {}, X_Test: {}, Y_Test: {}".format(len(x_train), len(y_train), len(x_test), len(y_test)))

    print('entrenando KNN ...')
    knn_model = trainKNN(X=x_train, y=y_train, model_save_path="trained_knn_model.clf", n_neighbors=5)
    print('... KNN entrenado')

