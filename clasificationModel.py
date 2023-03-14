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


if __name__ == "__main__":
    train_dir = "knn_examples/train"
    test_dir = "knn_examples/test"
    verbose = False

    X = []
    y = []

    print('transformando datos train de entrada ...')
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

    print('... datos train de entrada transformados')

    print(len(X))
    print(len(y))
    # Convertir X y y en un DataFrame de Pandas
    df = pd.DataFrame(data=X)
    df['etiquetas'] = y  # agregar las etiquetas como una columna adicional

    # Guardar el DataFrame como un archivo CSV
    df.to_csv('datos_train.csv', index=False)


    X = []
    y = []

    print('transformando datos test de entrada ...')
    # Iteramos para cada persona en el conjunto de entrenamiento
    for class_dir in os.listdir(test_dir):
        if not os.path.isdir(os.path.join(test_dir, class_dir)):
            continue

        # Iteramos para cada imagen de entrenamiento para la persona actual
        for img_path in image_files_in_folder(os.path.join(test_dir, class_dir)):
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

    print('... datos test de entrada transformados')

    print(len(X))
    print(len(y))
    # Convertir X y y en un DataFrame de Pandas
    df = pd.DataFrame(data=X)
    df['etiquetas'] = y  # agregar las etiquetas como una columna adicional

    # Guardar el DataFrame como un archivo CSV
    df.to_csv('datos_test.csv', index=False)

    print('particionando datos ...')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('... datos particionados')
    print("X_Train: {}, Y_Train: {}, X_Test: {}, Y_Test: {}".format(len(x_train), len(y_train), len(x_test), len(y_test)))


