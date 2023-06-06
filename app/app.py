from flask import Flask, render_template, request
from cluster_predictor import ClusterPredictor
from household_clustering import HouseholdClustering

app = Flask(__name__)

k = 7
transfer_df_income_path = 'transfer_df_income.pkl'
transfer_data_path = 'transfer_data.pkl'
cluster_predictor = ClusterPredictor(transfer_data_path, transfer_df_income_path, k)
cluster_describer = HouseholdClustering(transfer_data_path,transfer_df_income_path, k)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        cluster_number = cluster_predictor.predict(form_data)
        results = cluster_describer.summarize_cluster(cluster_number)
        return render_template('predict_results.html', results=results)

    return render_template('predict.html')

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    if request.method == 'POST':
        cluster_number = int(request.form.to_dict()['cluster-number'])
        results = cluster_describer.summarize_cluster(cluster_number)
        return render_template('clusters_info.html', results=results)

    return render_template('clusters.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
