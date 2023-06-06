# CS 180 project: Predictive model of household income using clustering

Jupyter notebook [link](https://colab.research.google.com/drive/1ZINpKCB4fgwIflOegEl8fXuPwnB5cwal?usp=sharing&authuser=1#scrollTo=XBLhtzz22Xw_)

Web app [link](https://cs180-project.onrender.com/)

## Description

In this project, a dataset based on the Family Income and Expenditure Survey (FIES) in the Philippines is put through K-means clustering. Through clustering, we aim to figure out possible classifications for households using the data from the dataset. We suspect that the clusters will be correlated with income or geographic areas from which a household belongs.

## Dev

- Clone the repository
```console
git clone https://github.com/rudnam/cs180-project.git
```
- Create environment
```console
cd cs180-project/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Start flask and visit `http://localhost:5000`
```console
python3 app.py
```

