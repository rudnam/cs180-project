from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the KMeans model from the pickle file
file_path = os.path.join(os.getcwd(), 'files', 'TEST_kmeans_model.pkl')
with open(file_path, 'rb') as file:
    kmeans_model = pickle.load(file)

# Load the KMeans clustering results from the pickle file
file_path = os.path.join(os.getcwd(), 'files', 'TEST_kmeans_clustering.pkl')
with open(file_path, 'rb') as file:
    kmeans_clustering = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        x = float(request.form.get('x'))
        y = float(request.form.get('y'))
        point = np.array([[x, y]])
        cluster_label = kmeans_model.predict(point)[0]
        result = {'x': x, 'y': y, 'cluster_label': cluster_label}
        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
