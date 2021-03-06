
from os import listdir, makedirs

import cv2
from numpy import savez_compressed, asarray
# face detection for the 5 Celebrity Faces Dataset
from os.path import isdir
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


def take_face(name):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    standard_name = "_".join(name.split(' '))
    path_train = 'data/train/' + standard_name + '/'
    path_val = 'data/val/' + standard_name + '/'
    makedirs(path_val, exist_ok=True)
    makedirs(path_train, exist_ok=True)
    sampleNum = 1


    while (True):
        # camera read
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = gray[y:y + h, x:x + w]
            face = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)

            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite(path_train + str(sampleNum) + ".jpg", face)
            if sampleNum > 10:
                cv2.imwrite(path_val + str(sampleNum) + ".jpg", face)

            cv2.imshow('frame', img)
        # wait for 100 miliseconds
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 20
        elif sampleNum > 16:
            break
        print(sampleNum)
    cam.release()
    cv2.destroyAllWindows()


def create_array_and_label():
    def load_faces(directory):
        faces = list()
        # enumerate files
        for filename in listdir(directory):
            # path
            path = directory + filename
            # get face
            face = cv2.imread(path)
            # store
            faces.append(face)
        return faces

    def load_dataset(directory):
        X, y = list(), list()
        # enumerate folders, on per class
        for subdir in listdir(directory):
            # path
            path = directory + subdir + '/'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)
    # load train dataset
    trainX, trainy = load_dataset('data/train/')
    # load test dataset
    testX, testy = load_dataset('data/val/')
    # save arrays to one file in compressed format
    savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def extract_embedding():
    # load the face dataset
    data = load('faces-dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    # load the facenet model
    model = load_model('facenet_keras.h5')
    print('Loaded Model')
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    # convert each face in the test set to an embedding
    print(trainy.shape)
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)

    print(newTestX.shape)
    # save arrays to one file in compressed format
    savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)


def predict():
    # train model
    data = load('faces-dataset.npz')
    testX_faces = data['arr_2']
    # load face embeddings
    data = load('faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    # take face and image has face
    image = list()
    face_img_list = list()
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    flag = True
    count = 0
    while(flag):
        count +=1
        print(count)
        ret, img = cam.read()
        print(000)
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = img[y:y + h, x:x + w]
            face = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
            face_img_list.append(face)
            image.append(img)
            cv2.imshow('frame', img)
            cv2.waitKey(1000)
        if len(face_img_list) == 1:
            flag = False
    img = image[0]
    face = face_img_list[0]

    facenet_model = load_model('facenet_keras.h5')
    embedding_pred_face = get_embedding(facenet_model, face)

    samples = expand_dims(embedding_pred_face, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100

    if class_probability > 70:
        print(222)
        predict_names = out_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        display_name = "".join(str(predict_names[0]).split("_"))
        cv2.putText(img, display_name+"    "+str(class_probability), (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow('frame', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit()
        cv2.destroyAllWindows()
    else:
        cv2.putText(img, "Not found user", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow('frame', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit()
        cv2.destroyAllWindows()


def save_data(name):
    take_face(name)
    create_array_and_label()
    extract_embedding()


