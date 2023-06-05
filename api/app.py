from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# # Load the KMeans model from the pickle file
# with open('_TEST_kmeans_model.pkl', 'rb') as file:
#     kmeans_model = pickle.load(file)

# # Load the KMeans clustering results from the pickle file
# with open('_TEST_kmeans_clustering.pkl', 'rb') as file:
#     kmeans_clustering = pickle.load(file)
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=3)

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Fit the model with the sample data
kmeans_clustering = kmeans_model.fit_predict(X)

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
