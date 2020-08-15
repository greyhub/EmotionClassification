from app import app
from flask import request, redirect, jsonify
from flask import render_template
from werkzeug.utils import secure_filename
import sys
import os
from pprint import pprint
import base64
import urllib.request
import json
import http.client
from cv2 import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import keras.backend as K

def swish_activation(x):
    return (K.sigmoid(x) * x)

app.config["IMAGE_STATIC"] = 'app/static/img/object_detected'
app.config["MODEL"] = 'app/core'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024


@app.route("/")
def index():
    return render_template("public/index.html")


@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>
    """


def allowed_image(filename):
    
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    
    if request.method == "POST":
        if request.files:
            img = request.files["image"]
            execution_path = os.getcwd()

            model_path = os.path.join(execution_path, app.config["MODEL"], 'model-dllab.h5')
            filename = secure_filename(img.filename)
            uploaded_path = os.path.join(execution_path, app.config["IMAGE_STATIC"], 'input', filename)
            img.save(uploaded_path)
            # model = load_model(model_path)
            model = tf.keras.models.load_model(model_path)

            # dimensions of our images
            img_width, img_height = 48, 48
            saved_img = image.load_img(uploaded_path, target_size=(img_width, img_height))
            # x = image.img_to_array(saved_img)
            saved_img = cv2.cvtColor(np.float32(saved_img), cv2.COLOR_RGB2GRAY)
            x = image.img_to_array(saved_img)
            x = x.reshape(48, 48, 1)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict_classes(images, batch_size=10)
            # classes = classes[0][0]
            if classes == 0:
                res = 'Angry'
            elif classes == 1:
                res = 'Disgust'
            elif classes == 2:
                res = 'Fear'
            elif classes == 3:
                res = 'Happy'
            elif classes == 4:
                res = 'Neutral'
            elif classes == 5:
                res = 'Sad'
            elif classes == 6:
                res = 'Surprise'


            K.clear_session()
            # return jsonify({
            #     'mes': 'ok',
            #     'result': res
            #     }), 201

            img_temp_path = '/static/img/object_detected/input/' + filename
            string_array = [res, img_temp_path]
            return render_template('public/upload_image.html', value=string_array)
        return jsonify({
            'result': 'error',
            'read_text': ''}), 201   

    return render_template("public/upload_image.html")

   
