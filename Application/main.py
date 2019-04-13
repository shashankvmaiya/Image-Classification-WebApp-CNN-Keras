from flask import Flask, render_template, request, url_for, jsonify
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import keras
from keras.preprocessing import image
import base64
import io

# from keras.models import load_model
# import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/predict', methods=['POST'])
@app.route("/", methods=['POST'])
def predict():
    model = keras.models.load_model("models/keras_mnist.h5")
    if request.method == 'POST':
        data = request.get_json()['data']
        data = base64.b64decode(data)
        img_data = io.BytesIO(data)
        img = image.load_img(img_data, target_size=(28, 28))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        digit = np.argmax(predictions)
    return render_template('results.html', prediction=digit)
#     return jsonify({"prediction": data})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
