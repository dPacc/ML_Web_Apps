from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv("data/YouTubeSpamMergedData.csv")
    df_data = df[["CONTENT", "CLASS"]]
    # # Features and Labels
    X = df_data['CONTENT']
    y = df_data['CLASS']
    # # Extract feature with CountVectorizer
    corpus = X
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    # # Naive Bayes Classifier
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    # classifier = MultinomialNB()
    # classifier.fit(X_train, y_train)
    # classifier.score(X_test, y_test)
    # Alternative usage of model
    ytbmodel = open("data/YTSpamModel.pkl", "rb")
    classifier = joblib.load(ytbmodel)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)

    return render_template('result.html', prediction = my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
