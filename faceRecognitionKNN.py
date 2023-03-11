from sklearn import neighbors
import math
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# entrena un clasificadro k vecinos mas crecanos para reconocimiento facial
def train(train_dir, model_save_path=None, n_neighbors=None, km_algo='ball_tree', verbose=False):
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


# reconoce una imagen dadda usando un clasificadro KNN entrenado 0.47
def predict(X_img_path, knn_clsf=None, model_path=None, distance_threshold=0.47):
    # X_img_path: direccion de la imagen a reconocer
    # knn_clf: un objeto clasificador knn. si no se especifica, se debe especificar model_save_path.
    # model_path: camino a un clasificador knn pre entrnado. si no se especifica, model_save_path debe ser knn_clsf.
    # distance_threshold: Umbral de distancia para la clasificación de rostros. cuanto más grande es, más posibilidades
    #                     de clasificar erróneamente a una persona desconocida como conocida.
    # retornamos una lista de nombres y locaciones de caras para las caras reconocidas en la imagen
    # para las caras no reconocidas se retorna el nombre de "unknown"

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
    print(faces_encodings)
    # usamos el modelo KNN para encontrar las mejores coincidencias para la iamgen de test
    closest_distances = knn_clsf.kneighbors(faces_encodings, n_neighbors=1)
    print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(are_matches)

    # predecimos las clases y removemos las clasificaiones que no estan en el humbral
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clsf.predict(faces_encodings), X_face_locations, are_matches)]


# mostramos las predicciones en la imagen
def show_prediction_labels_on_image(img_path, predictions):
    # img_path: direccion de la imagen a reconocer
    # predictions: resultados de la prediccion de la funcion

    # transformamos la iamgen a RGB y la mostramos
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Ddibujamos un cuadrado al rededor de la cara
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # reajustamos el formato del nombre
        name = name.encode("UTF-8")

        # mostramos el nombre
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # eliminamos la libreria de dibujo de la memoria segun la documentacion de Pillow
    del draw

    # Mostramos el resultado de la imagen
    pil_image.show()
    img_dir = "knn_examples/result"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_filename = "imagen_con_predicciones.jpg"
    img_path = os.path.join(img_dir, img_filename)
    pil_image.save(img_path)


def face_train():
    print("Entrenando clasificador KNN...")
    train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=5)
    print("Entrenamiento Completado!")


def face_rec(path, name_predict):
    for image in os.listdir(path):
        full_file_path = os.path.join(path, image)

        print("Buscando caras en: {}".format(image))

        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        print(predictions)

        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
            if name == name_predict:
                return True
    return False


if __name__ == "__main__":
    # Paso1 : entrenamos el clasificadro y lo guardamos en el disco
    # una vez que este entrenado el modelo se puede omitir este paso.
    '''print("Entrenando clasificador KNN...")
    #classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=5)
    print("Entrenamiento Completado!")

    # Paso 2: utilizamos el clasificador entrenado, para predecir las imagenes desconocidas
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)
        print(full_file_path)
        print("Buscando caras en: {}".format(image_file))

        # encontramos todas las personas en la imagen utilizando el modelo de clasificacion entrenado
        # Nota: se puede utilizar un archivo pre-entrenado o una instancia al metodo de entrenamiento
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        print(predictions)

        # imprimimos los resultados en la consola
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # desplegamos los resultados de la imagen
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
    '''

    validation = face_rec("knn_examples/test", 'Ismael')
    print(validation)
