from flask import Flask, render_template, request
from predict import ModelWrapper
import pickle
import numpy as np

app = Flask(__name__)

# Load the KMeans model from the pickle file
with open('TEST_kmeans_model.pkl', 'rb') as file:
    test_kmeans_model = pickle.load(file)

# Load the KMeans clustering results from the pickle file
with open('TEST_kmeans_clustering.pkl', 'rb') as file:
    test_kmeans_clustering = pickle.load(file)

model_path = 'kmeans_model.pkl'
clustering_path = 'kmeans_clustering.pkl'

model_wrapper = ModelWrapper(model_path, clustering_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        x = float(request.form.get('x'))
        y = float(request.form.get('y'))
        point = np.array([[x, y]])
        cluster_label = test_kmeans_model.predict(point)[0]
        result = {'x': x, 'y': y, 'cluster_label': cluster_label}
        return render_template('index.html', result=result)

    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        dict_data = request.form.to_dict()
        print(dict_data)
        result = model_wrapper.predict(dict_data)
        return render_template('predict.html', result=result)

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
