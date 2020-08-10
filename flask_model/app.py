from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flasgger import Swagger

with open("/var/www/flask_model/rf.pkl", "rb") as model_pkl:
    model = pickle.load(model_pkl)

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    # a = request.form["a"]
    # b = request.form["b"]
    return "Hello World!"


@app.route("/predict", methods=['POST'])
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: formData
        type: number
        required: true
      - name: s_width
        in: formData
        type: number
        required: true
      - name: p_length
        in: formData
        type: number
        required: true
      - name: p_width
        in: formData
        type: number
        required: true
    responses:
     200:
       description: prediction for iris
       schema:
         id: IRIS
         properties:
           prediction:
             type: string
             description: prediction for iris
             default: ""
    """
    s_length = request.form["s_length"]
    s_width = request.form["s_width"]
    p_length = request.form["p_length"]
    p_width = request.form["p_width"]

    prediction = model.predict(np.array([[s_length, s_width,
                                          p_length, p_width]]))
    # return {"result": str(prediction)}
    return jsonify(str(prediction))


@app.route("/predict_file", methods=['POST'])
def predict_iris_file():
    """Example file endpoint returning a list of predictions of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses:
         200:
           description: A list of predictions
           schema:
             id: IRIS
             properties:
               prediction:
                 type: string
                 description: The list of predictions
                 default: []
    """
    input_data = pd.read_csv(request.files["input_file"], header=None)
    prediction = model.predict(input_data)
    return jsonify(str(list(prediction)))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
