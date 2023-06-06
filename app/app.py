from flask import Flask, render_template, request
from cluster_predictor import ClusterPredictor
import pickle
import numpy as np

app = Flask(__name__)

transfer_df_income_path = 'transfer_df_income.pkl'
transfer_data_path = 'transfer_data.pkl'
cluster_predictor = ClusterPredictor(transfer_data_path, transfer_df_income_path, 7)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        print(form_data)
        result = cluster_predictor.predict(form_data)
        return render_template('predict.html', result=result)

    return render_template('predict.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
