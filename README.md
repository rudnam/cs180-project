# CS 180 project: Clustering of Filipino Households using K-Means

Data set [https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure](https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure)

Google colab [https://colab.research.google.com/drive/13gjczh-z4OHZHS1QJRjv6Qr5TqE3LhEj?usp=sharing](https://colab.research.google.com/drive/13gjczh-z4OHZHS1QJRjv6Qr5TqE3LhEj?usp=sharing)

Web app [https://cs180-project.onrender.com/](https://cs180-project.onrender.com/)

## Description

This project involves analyzing a dataset derived from the Family Income and Expenditure Survey (FIES) in the Philippines using K-means clustering. The goal is to identify potential household classifications by leveraging the dataset's information. Our hypothesis suggests that these classifications will likely be associated with income levels and geographic regions. 

## Dev

Clone the repository

```console
git clone https://github.com/rudnam/cs180-project.git
```

Create environment

```console
cd cs180-project/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start flask and visit `http://localhost:5000`

```console
python3 app.py
```

