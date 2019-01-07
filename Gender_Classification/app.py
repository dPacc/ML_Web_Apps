from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/names_dataset.csv")
    # Features and Labels
    df_X = df.name
    df_y = df.sex

    corpus = df_X
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)

    DTM = open("models/naivebayesgendermodel.pkl", "rb")
    classifier = joblib.load(DTM)

    if request.method == 'POST':
        name = request.form['namequery']
        data = [name]
        vect = cv.transform(data).toarray()
        my_pred = classifier.predict(vect)

    return render_template('results.html', pred = my_pred, name = name.upper())


if __name__ == "__main__":
    app.run(debug=True)
