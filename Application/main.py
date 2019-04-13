from flask import Flask, request, jsonify
import numpy as np
import keras
import matplotlib

#from keras.models import load_model
#import matplotlib.pyplot as plt

app = Flask(__name__)
model = keras.models.load_model("keras_mnist.h5")
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.data['input']
        #img = matplotlib.pyplot.plt.imread(data)
        #result = model.predict(img)
        #digit = np.argmax(result)
    return jsonify({"prediction": data})


