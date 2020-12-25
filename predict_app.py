import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request,jsonify, render_template

app=Flask(__name__)

def getModel():
    global model
    model=load_model('MobileNet-Tuned-Cats-Dogs.h5')
    print("Model Loaded")

def preProcessImage(image,targetSize):
    if image.mode !='RGB':
        image=image.convert('RGB')
    image=image.resize(targetSize)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)

    return tf.keras.applications.mobilenet.preprocess_input(image)

print("Loading Keras Model")
getModel()

@app.route("/")
def index():
    return render_template("predict.html")


@app.route("/predict",methods=['POST'])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processedImage=preProcessImage(image,targetSize=(224,224))
    prediction=model.predict(processedImage).tolist()
    print(prediction)
    response={
        'prediction':{
            'cat':prediction[0][0],
            'dog':prediction[0][1]
        }
    }

    return jsonify(response)

app.run(debug=True)
