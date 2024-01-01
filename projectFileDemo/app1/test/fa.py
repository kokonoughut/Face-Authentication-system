import cv2
import os
import tensorflow
import numpy as np

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model,model_from_json



def login_with_image():
    model_path = "E:\zeck\Face_Recognition_using_Siamese_Network-master\Models\Inception_ResNet_v1.json"
    weights_path = r"E:\zeck\Face_Recognition_using_Siamese_Network-master\Models\facenet_keras_weights.h5"
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    enc_model = model_from_json(loaded_model_json)
    enc_model.load_weights(weights_path)
    global label
    mtcnn_detector = MTCNN()
        # Function to detect and extract face from a image
    def detect_face(filename, required_size=(160, 160),normalize = True):
        img = Image.open(filename)
        # convert to RGB
        img = img.convert('RGB')

        # convert to array
        pixels = np.asarray(img)

        # detect faces in the image
        results = mtcnn_detector.detect_faces(pixels)

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        if normalize == True:
            mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
            std = np.std(face_array, axis=(0,1,2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            return (face_array - mean) / std
        else : 
            return face_array
    # Compute Face encodings and load IDs of known persons
    # Update face database path according to your working environment
    known_faces_encodings = []
    known_faces_ids = []
    known_faces_path = "test\Face_database"
    for filename in os.listdir(known_faces_path):

      # Detect faces
      face = detect_face(known_faces_path+filename,normalize = True)
      # Compute face encodings
      feature_vector = enc_model.predict(face.reshape(1,160,160,3))
      feature_vector/= np.sqrt(np.sum(feature_vector**2))
      known_faces_encodings.append(feature_vector)
      # Save Person IDs
      label = filename.split('.')[0]
      known_faces_ids.append(label)
    known_faces_encodings = np.array(known_faces_encodings).reshape(len(known_faces_encodings),128)
    known_faces_ids = np.array(known_faces_ids)
    # Function to recognize a face (if it is in known_faces)
    def recognize(img,known_faces_encodings,known_faces_ids,threshold = 0.75):
      scores = np.zeros((len(known_faces_ids),1),dtype=float)
      enc = enc_model.predict(img.reshape(1,160,160,3))
      enc/= np.sqrt(np.sum(enc**2))
      scores = np.sqrt(np.sum((enc-known_faces_encodings)**2,axis=1))
      match = np.argmin(scores)
      if scores[match] > threshold :
        return ("UNKNOWN",0)
      else :
        return (True)
    # Fnction to perform real-time face recognition through a webcam
    def face_recognition(mode,file_path,known_faces_encodings,known_faces_ids,
                             detector = 'haar', threshold = 0.75):
      if detector == 'haar':
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('app1/Models/haarcascade_frontalface_default.xml')
      if mode == 'webcam':
        # To capture webcam feed. Change argument for differnt webcams
        cap = cv2.VideoCapture(0)
      elif mode == 'video':
        # To capture video feed 
        cap = cv2.VideoCapture(file_path)
      while True:
        # Read the frame
        _, img = cap.read()
        # Stop if end of video file
        if _ == False:
            break;
        if detector == 'haar':
        
          #Convert to grayscale
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          # Detect the faces
          faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        elif detector == 'mtcnn' :  
          results = mtcnn_detector.detect_faces(img)
          if(len(results)==0):
            continue
          faces = []
          for i in range(len(results)):
            x,y,w,h = results[i]['box']
            x, y = abs(x), abs(y)
            faces.append([x,y,w,h])
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            image = Image.fromarray(img[y:y+h, x:x+w])
            image = image.resize((160,160))
            face_array = asarray(image)
            # Normalize
            mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
            std = np.std(face_array, axis=(0,1,2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            face_array_normalized = (face_array - mean) / std
            # Recognize
            label1 = recognize(face_array_normalized,known_faces_encodings,known_faces_ids,threshold = 0.75)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            #cv2.putText(img, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        # Display
        cv2.imshow('Face_Recognition', img)
        # Stop if escape key is pressed

        if cv2.waitKey(1) and  label1 is True:
          cap.release() 
          cv2.destroyAllWindows()  
          return True
        elif cv2.waitKey(1) & 0XFF == ord('q'):
          cap.release() 
          cv2.destroyAllWindows()  
          return False

          
    a=face_recognition('webcam',None,known_faces_encodings,known_faces_ids,
             detector = 'haar',threshold = 0.75)

    print(a) 
