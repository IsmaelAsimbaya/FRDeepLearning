import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
import re, json
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import os
from capturadorRostrosB64 import video_capture
from faceRecognitionKNN import face_rec, redimension, face_train
from PIL import Image
import io
import datetime
from flask import send_file
import requests

app = Flask(__name__)

cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


@app.route('/save-user', methods=['POST'])
def face_recognition():
    # obtener los datos del POST
    nombre = request.json['nombre']
    identificacion = request.json['identificacion']
    correo = request.json['correo']
    telefono = request.json['telefono']
    firma = request.json['firma']
    base64_string = request.json['video']
    entrada = request.json['entrada']
    salida = request.json['salida']

    # guardar los datos en Firestore
    doc_ref = db.collection('users').document()
    doc_ref.set({
        'nombre': nombre,
        'identificacion': identificacion,
        'correo': correo,
        'telefono': telefono,
        'firma': firma,
        'entrada': entrada,
        'salida': salida
    })
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(script_dir, 'videos', 'train', identificacion)
    filename = identificacion + ".mp4"

    video_bytes = base64.b64decode(base64_string)
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    with open(os.path.join(dataPath, filename), "wb") as video_file:
        video_file.write(video_bytes)

    return 'Datos guardados exitosamente'


@app.route('/edit-user/<id>', methods=['PUT'])
def edit_user(id):
    # obtener los datos del POST
    nombre = request.json['nombre']
    identificacion = request.json['identificacion']
    correo = request.json['correo']
    telefono = request.json['telefono']
    firma = request.json['firma']
    entrada = request.json['entrada']
    salida = request.json['salida']

    # entrada_obj = datetime.strptime(entrada, '%Y-%m-%dT%H:%M:%S.%fZ')
    # salida_obj = datetime.strptime(salida, '%Y-%m-%dT%H:%M:%S.%fZ')

    # guardar los datos en Firestore
    doc_ref = db.collection('users').document(id)
    doc_ref.update({
        'nombre': nombre,
        'identificacion': identificacion,
        'correo': correo,
        'telefono': telefono,
        'firma': firma,
        'entrada': 1633210800,
        'salida': 1633210800
    })

    return 'Datos guardados exitosamente'


@app.route('/', methods=['GET'])
def get_users():
    docs = db.collection('users').get()
    users = []
    for doc in docs:
        user = doc.to_dict()
        user['id'] = doc.id
        user['entrada'] = "2023-03-13T10:00:00"
        user['salida'] = "2023-03-13T10:00:00"
        users.append(user)

    return jsonify(users)


@app.route('/aprender/<id>', methods=['GET'])
def aprender(id):
    docs = db.collection('users').where('identificacion', '==', id).get()
    if len(docs) == 0:
        return 'Usuario no encontrado'
    else:
        user = docs[0].to_dict()
        user['id'] = docs[0].id
        identificacion = user['identificacion']
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataPath = os.path.join(script_dir, 'videos', 'train', identificacion)
        filename = identificacion + ".mp4"
        with open(dataPath + '/' + filename, 'rb') as archivo:
            # Leer el contenido del archivo y convertirlo a base64
            contenido_base64 = base64.b64encode(archivo.read())
        cadena_base64 = contenido_base64.decode('utf-8')
        saved = video_capture(user['identificacion'], cadena_base64)
        face_train()
        if saved:
            db.collection('users').document(user['id']).update({
                'aprendido': True
            })
        return jsonify(user)


@app.route('/validar', methods=['POST'])
def validar():
    # obtener los datos del POST
    imageB64 = request.json['imageB64']
    identificacion = request.json['identificacion']
    image_data = base64.b64decode(imageB64)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    now = datetime.datetime.now()
    fecha_hora = now.strftime("%Y%m%d%H%M%S")
    image_date_name = 'val_' + identificacion + '_' + fecha_hora
    dataPath = os.path.join(script_dir, 'knn_examples', 'val', identificacion, image_date_name + '.jpg')
    if not os.path.exists(os.path.join(script_dir, 'knn_examples', 'val', identificacion)):
        os.makedirs(os.path.join(script_dir, 'knn_examples', 'val', identificacion))
    with open(dataPath, 'wb') as archivo:
        archivo.write(image_data)

    return face_rec(redimension(dataPath), identificacion)


@app.route('/descargar-contrato')
def mostrar_pdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(script_dir, 'ContratoFirmado.pdf')
    return send_file(dataPath, as_attachment=True)


def firmar():
    valor = os.getenv('contraseña_firma', 'Pin2021**')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # base 64
    with open("Contrato.pdf", "rb") as archivo:
        # Leer los datos del archivo
        datos = archivo.read()
        # Codificar los datos en Base64
        datos_codificados = base64.b64encode(datos)
    with open("firma.pfx", "rb") as archivo:
        # Leer los datos del archivo
        datosfirma = archivo.read()
        # Codificar los datos en Base64
        datos_codificados_firma = base64.b64encode(datosfirma)

    datos = {
        "bufferSignature": datos_codificados_firma.decode('utf-8'),
        "b64Pdf": datos_codificados.decode('utf-8'),
        "password": "Pin2021**",
        "options": {
            "x": "50",
            "y": "50",
            "page": "2",
            "type": "personal",
            "reason": "Firma de entrada al trabajo: TEST",
        }
    }
    datos = json.dumps(datos)
    token = os.getenv('token')
    if token is None:
        return "false"
    headers = {'Authorization': token,
               'Content-Type': 'application/json'}
    response = requests.post("https://api-firmado.pdfecuador.com/api/signatures/sign", data=datos, headers=headers)
    print(response.content)
    if response.status_code != 200:
        return "false"
    newbase64 = response.json()['b64']
    newbase64 = newbase64.encode('utf-8')
    newbase64 = base64.b64decode(newbase64)

    dataPath = os.path.join(script_dir, 'ContratoFirmado.pdf')
    with open(dataPath, 'wb') as archivo:
        archivo.write(newbase64)

    return "true"


app.run(host='0.0.0.0', port='5001', debug=True)
