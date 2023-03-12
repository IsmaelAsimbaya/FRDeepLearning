import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
import re, json
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import os
from capturadorRostrosB64 import video_capture
from faceRecognitionKNN import face_rec, redimension
from PIL import Image
import io

app = Flask(__name__)

cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def print_request(request):
    # Print request url
    print(request.url)
    # print relative headers
    print('content-type: "%s"' % request.headers.get('content-type'))
    print('content-length: %s' % request.headers.get('content-length'))
    # print body content
    body_bytes = request.get_data()
    # replace image raw data with string '<image raw data>'
    body_sub = re.sub(b'(\r\n\r\n)(.*?)(\r\n--)', br'\1<image raw data>\3', body_bytes, flags=re.DOTALL)
    print(body_sub.decode('utf-8'))


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
    dataPath = os.path.join(script_dir, 'videos', 'usuarios', identificacion)
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

    entrada_obj = datetime.strptime(entrada, '%Y-%m-%dT%H:%M:%S.%fZ')
    salida_obj = datetime.strptime(salida, '%Y-%m-%dT%H:%M:%S.%fZ')

    # guardar los datos en Firestore
    doc_ref = db.collection('users').document(id)
    doc_ref.update({
        'nombre': nombre,
        'identificacion': identificacion,
        'correo': correo,
        'telefono': telefono,
        'firma': firma,
        'entrada': entrada_obj.timestamp(),
        'salida': salida_obj.timestamp()
    })

    return 'Datos guardados exitosamente'


@app.route('/', methods=['GET'])
def get_users():
    docs = db.collection('users').get()
    users = []
    for doc in docs:
        user = doc.to_dict()
        user['id'] = doc.id
        user['entrada'] = datetime.fromtimestamp(user['entrada']).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        user['salida'] = datetime.fromtimestamp(user['salida']).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
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
        dataPath = os.path.join(script_dir, 'videos', 'usuarios', identificacion)
        filename = identificacion + ".mp4"
        with open(dataPath + '/' + filename, 'rb') as archivo:
            # Leer el contenido del archivo y convertirlo a base64
            contenido_base64 = base64.b64encode(archivo.read())
        cadena_base64 = contenido_base64.decode('utf-8')
        saved = video_capture(user['identificacion'], cadena_base64)
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
    video_data = base64.b64decode(imageB64)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        image = Image.open(io.BytesIO(video_data))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataPath = os.path.join(script_dir, 'knn_examples', 'val', identificacion, identificacion + '.jpg')
        with open(dataPath, 'wb') as f:
            image.save(f)

    return 'true' if face_rec(redimension(dataPath), identificacion) else 'false'


app.run(host='0.0.0.0', port='5001', debug=True)
