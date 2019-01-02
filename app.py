from flask import Flask, render_template, flash, redirect, url_for, session, logging, request
from wtforms import Form, StringField, validators
import Project

import re

app = Flask(__name__)


@app.route("/search")
def search():
    return render_template('search.html')


class WordPredictionForm(Form):
    word = StringField('', [validators.Length(min=1, max=1000)])

# PROJECT NLP

@app.route('/', methods=['GET', 'POST'])
def index():
    form = WordPredictionForm(request.form)
    if request.method == 'POST' and form.validate():
        word = form.word.data
        print(word)


        #Predict the Model
        project = Project

        word = re.sub(r'([^\s\w]|_)+', '', word)
        seq = word[:40].lower()
        # print(seq)
        list = project.predict_completions(seq, 5)
        chosen = list[0]

        print(list)

        flash("loading...")

        # redirect(url_for('index', list=list))
        return render_template('index.html', form=form, list=list, seq=seq, chosen=chosen, scroll='result')
    return render_template('index.html', form=form)



if __name__ == "__main__":
    app.secret_key = "secret123"
    app.run(debug=True)
