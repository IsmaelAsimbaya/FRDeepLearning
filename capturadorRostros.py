import cv2
import os
import imutils


def video_capture(personName):
    dataPath = 'C:/HardDisk/Biblioteca/Workspaces/Python/FRDeepLearning/knn_examples/test'
    videoPath = 'C:/HardDisk/Biblioteca/Workspaces/Python/FRDeepLearning/videos/test'
    personPath = dataPath + '/' + personName

    if not os.path.exists(personPath):
        print('Directorio creado: ', personPath)
        os.makedirs(personPath)
    cap = cv2.VideoCapture(videoPath + '/' + personName + '.mp4')
    # cap = cv2.VideoCapture(0)

    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.12,
                                             # scaleFactor=1.3,
                                             minNeighbors=5,
                                             minSize=(200, 200),
                                             maxSize=(450, 450))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_capture('3')
