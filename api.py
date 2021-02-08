from flask import Flask
from flask import request
from flask import render_template

import numpy as np 
import matplotlib.pyplot as plt
import torch 

import albumentations as A
import onnxruntime
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = "/home/naim/Desktop/Projects/masked-face-recognition/static"

class_dict = {4:'mindy_kaling' , 1 :'elton_john' , 0 : 'ben_afflek' ,2: 'jerry_seinfeld' ,  3 : 'madonna'}

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def sigmod(x) : 
    return 1/(1+ np.exp(-x))

class Predicator() : 
    def __init__(self , mask_detector_path , unmasked_recognizer_path , masked_recognizer_path) : 
        self.mask_detector          = onnxruntime.InferenceSession(mask_detector_path)
        self.unmasked_recognizer    = onnxruntime.InferenceSession(unmasked_recognizer_path)
        self.masked_recognizer = onnxruntime.InferenceSession(masked_recognizer_path)
    def prep_image(self , image ):
        aug = A.Compose([
                              A.Resize(224, 224, p= 1.0),
                              A.Normalize(
                                  mean=[0.485],
                                  std=[0.229],
                                  max_pixel_value=255.0,
                                  p=1.0,
                              ),
                          ],
                            p=1.0,
                        )

        image = aug(image  = np.array(image))['image']
        image = np.transpose(image , (2,0,1)).astype(float) 
        image = torch.tensor(image ,dtype = torch.float).unsqueeze(0)
        return image
    
    def batch_pred(self , images) : 
        all_images = [] 
        for image in images : 
            all_images.append(self.prep_image(image))
        all_images = torch.cat(all_images , dim = 0)
        inputs = {self.mask_detector.get_inputs()[0].name: to_numpy(all_images)}
        outs = self.mask_detector.run(None, inputs)
        outs = sigmod(outs[0][0])
        if outs > 0.5 : 
            outs = self.masked_recognizer.run(None, inputs)
            predicted_class = softmax(outs[0]).argmax(axis=1)
            predicted_class = class_dict[predicted_class[0]]
            mask_prob = 'Masked'
            
        else : 
            outs = self.unmasked_recognizer.run(None, inputs)
            predicted_class = softmax(outs[0]).argmax(axis=1)
            predicted_class = class_dict[predicted_class[0]] 
            mask_prob = 'Unmasked'
            
        return  predicted_class,mask_prob
    
    def predict(self , image ) :
        image = self.prep_image(image)
        inputs = {self.mask_detector.get_inputs()[0].name: to_numpy(image)}
        outs = self.mask_detector.run(None, inputs)
        mask_prob = sigmod(outs[0])
        
        if outs > 0.5 : 
            outs = self.masked_recognizer.run(None, inputs)
            predicted_class = softmax(outs).argmax(axis=1)
        else : 
            outs = self.unmasked_recognizer.run(None, inputs)
            predicted_class = softmax(outs).argmax(axis=1)
        return  predicted_class,mask_prob

def detect_and_predict_mask(image_path, faceNet, predicator):
    frame =cv2.imread(image_path)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
         predicted_class,mask_prob = predicator.batch_pred(faces)
    return  predicted_class,mask_prob

ap = argparse.ArgumentParser()

ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")

ap.add_argument("-m", "--mask-detector", type=str,
    default="mask_detector.onnx",
    help="path to trained face mask detector model")

ap.add_argument("-d", "--masked-recognizer", type=str,
    default="masked_face_recognizer.onnx",
    help="path to trained face mask detector model")

ap.add_argument("-e", "--unmasked-recognizer", type=str,
    default="unmasked_face_recognizer.onnx",
    help="path to trained face mask detector model")

args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
mask_detector_path = args['mask_detector']

# load the unmasked face recognizer model from disk
print("[INFO] loading unmasked face recognizer model...")
unmasked_recognizer_path = args['unmasked_recognizer']

# load the masked face recognizer model from disk
print("[INFO] loading masked face recognizer model...")
masked_recognizer_path = args['masked_recognizer']
predicator = Predicator(mask_detector_path , unmasked_recognizer_path , masked_recognizer_path)


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)

            predicted_celeb , mask_status  = detect_and_predict_mask(image_location, faceNet, predicator)
            return render_template("index.html", predicted_celeb = predicted_celeb ,mask_pred = mask_status , image_loc=image_file.filename)

    return render_template("index.html", prediction=0, image_loc=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)