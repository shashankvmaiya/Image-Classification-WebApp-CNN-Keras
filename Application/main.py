from flask import Flask, render_template, request, url_for, jsonify

import numpy as np
import keras
from keras.preprocessing import image
import matplotlib

# from keras.models import load_model
# import matplotlib.pyplot as plt

app = Flask(__name__)

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path

@app.route("/")
def index():
    return render_template("index.html")

# @app.route("/", methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    model = keras.models.load_model("keras_mnist.h5")
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        # data = request.data['input']
        # img = matplotlib.pyplot.plt.imread(data)
        # result = model.predict(img)
        # digit = np.argmax(result)
    return render_template('results.html', prediction=data, comment=comment)
#     return jsonify({"prediction": data})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
