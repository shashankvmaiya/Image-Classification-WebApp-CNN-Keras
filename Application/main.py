from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__)
model = load_model("keras_mnist.h5")
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.data['input']
        #img = plt.imread(data)
        #result = model.predict(img)
        #digit = np.argmax(result)
    return jsonify({"prediction": data})


