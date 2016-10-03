from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from flask_sqlalchemy import SQLAlchemy
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect
# import update function from local dir
# from update import update_model

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
 'pkl_obj/classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'rtDiabetes.sqlite')
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
    c.execute("INSERT INTO rtDiabetes_db (tweet, rtBinary, date)" \
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=10, max=140)])

@app.route('/')
@app.route('/index')
@app.route('/input')
def input():
    form = ReviewForm(request.form)
    return render_template('input.html', form=form)

@app.route('/about')
def about():
    form = ReviewForm(request.form)
    return render_template('about.html', form=form)


@app.route('/contact')
def contact():
    form = ReviewForm(request.form)
    return render_template('contact.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)

        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=int(round(proba*100)))
    return render_template('input.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'No': 0, 'Yes': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
#    update_model(filepath=db, model=clf, batch_size=1000)
