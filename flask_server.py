from flask import Flask, request
import re, json
from faceRecognitionKNN import predict

app = Flask(__name__)


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


@app.route('/face_rec', methods=['POST', 'GET'])
def face_recognition():
    if request.method == 'POST':
        print_request(request)
        # check if the post request has the file part
        if 'file' in request.files:
            file = request.files.get('file')
            print(file)
            names = predict(file, model_path="trained_knn_model.clf")
            resp_data = {'name': names}
            return json.dumps(resp_data)

    return '''
    <!doctype html>
    <title>Face Recognition</title>
    <h1>Upload an image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


app.run(host='0.0.0.0', port='5001', debug=True)