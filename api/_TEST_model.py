# Make test model
from sklearn.cluster import KMeans
import numpy as np
import pickle

TEST_kmeans_model = KMeans(n_clusters=3)

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Fit the model with the sample data
TEST_kmeans_clustering = TEST_kmeans_model.fit_predict(X)

# Save the KMeans model to a pickle file
with open('api/_TEST_kmeans_model.pkl', 'wb') as file:
    pickle.dump(TEST_kmeans_model, file)

# Save the KMeans clustering results to a pickle file
with open('api/_TEST_kmeans_clustering.pkl', 'wb') as file:
    pickle.dump(TEST_kmeans_clustering, file)