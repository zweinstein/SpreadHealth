from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from flask_sqlalchemy import SQLAlchemy
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect
## import update function if I want to update the classifier training with new tweets
# from update import update_model

app = Flask(__name__)

######## Loading the Classifier and Connecting to SQLite Database
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_obj/classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'health.sqlite')

def classify(document):
    label = {0: 'No', 1: 'Yes'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO health_db (tweet, rtBinary, date)" \
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## Flask
class TweetForm(Form):
    tweet = TextAreaField('', [validators.DataRequired(), validators.length(min=10, max=140)])

@app.route('/')
@app.route('/index')
@app.route('/input')
def input():
    form = TweetForm(request.form)
    return render_template('input.html', form=form)

@app.route('/slides')
def slides():
    return render_template('slides.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/results', methods=['POST'])
def results():
    form = TweetForm(request.form)
    if request.method == 'POST' and form.validate():
        tweet = request.form['tweet']
        y, proba = classify(tweet)
        return render_template('results.html',
                                content=tweet,
                                prediction=y,
                                probability=int(round(proba*100)))
    return render_template('input.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    tweet = request.form['tweet']
    prediction = request.form['prediction']

    inv_label = {'No': 0, 'Yes': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(tweet, y)
    sqlite_entry(db, tweet, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
#    update_model(filepath=db, model=clf, batch_size=1000)
