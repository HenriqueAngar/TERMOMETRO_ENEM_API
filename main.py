import math

import keras.models as km
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


def process_requ(inputs):

    data = pd.DataFrame.from_dict(inputs, orient='index')
    data = data.T

    column_names = ['TP_SEXO', 'TP_COR_RACA', 'TP_ESCOLA',
                    'TP_LINGUA', 'RENDA', 'PAIS', 'CASA', 'INFO', 'FOCO']
    conversion_columns = ['RENDA', 'PAIS', 'CASA', 'INFO', 'FOCO']

    for name in column_names:

        data[name] = pd.to_numeric(data[name], errors='coerce')
        value = data.at[0, name]

        if math.isnan(value) or value < 0 or value > 100:
            data[name] = [0.5]

        elif name in conversion_columns:
            data[name] = data[name].values/100

    return data


model = km.load_model('model.h5')


def calc_nota(data):

    n = []
    values = model.predict(data)
    values = values[0]

    i = 0
    for value in values:
        n.append(round(value))
        i += 1

    notas = {'ch': n[0], 'cn': n[1], 'lc': n[2], 'mt': n[3], 'red': n[4]}
    return notas


app = Flask("termometro_enem")
cors = CORS(app, resources={

            '/predict/*': {"origins": 'https://henriqueangar.github.io'}})


@app.route('/', methods=['GET'])
def welcome():
    html_content = "<html><body><h1>Termômetro do Enem Online</h1><br><br><h2>Nada para ver aqui!</h2></body></html>"
    return html_content


@app.route('/predict', methods=['GET'])
def predict_nota():

    try:
        inputs = request.args.to_dict()
        info = process_requ(inputs)
        response = jsonify(calc_nota(info))

        response.status_code = 200
        response.headers.add("Access-Control-Allow-Origin",
                             'https://henriqueangar.github.io')
        response.headers.add('Content-Security-Policy',
                             "upgrade-insecure-requests")
        return response

    except Exception as e:
        response_error = {'error: ': str(e)}
        return response_error, 500


app.run(host="0.0.0.0")
