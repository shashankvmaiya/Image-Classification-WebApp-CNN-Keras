from flask import Flask, render_template, request, url_for, jsonify
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import base64
import io
import os
import sys
from werkzeug.utils import secure_filename


# from keras.models import load_model
# import matplotlib.pyplot as plt

app = Flask(__name__)

print('About to load the model')
MODEL_PATH = 'models/keras_mnist.h5'
model = load_model(MODEL_PATH)
print('Model loaded')


def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    print('Saved image at path: ', file_path)
    return file_path


@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/predict', methods=['POST'])
# @app.route("/", methods=['POST'])
@app.route('/predictResNet50', methods=['GET', 'POST'])
def predictResNet50():
    print('----------Entered predictResNet50---------')
    # basepath = os.path.dirname(__file__)
    # model_path = os.path.join(basepath, 'keras_mnist.h5')
    # print('Model Path: ', model_path)
    # model = keras.models.load_model(model_path)
    # print('Model loaded')
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        # data = request.get_json()['data']
        # data = base64.b64decode(data)
        # img_data = io.BytesIO(data)
        img = image.load_img(file_path, target_size=(28, 28), grayscale=True)
        x = (255-image.img_to_array(img))/255.0
        x = np.expand_dims(x, axis=0)
        print('Image Post-processing complete')
        predictions = keras_mnist.predict(x)
        print('Predictions completed, llrs = ', predictions)
        digit = str(np.argmax(predictions))
        print('Predicted digit = ', digit)
        return digit
    # return render_template('results.html', prediction=digit)
#     return jsonify({"prediction": data})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
