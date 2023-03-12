import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
import pickle, json, random
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

with open("sentiment_model.pkl", 'rb') as file:
    model = pickle.load(file)

with open("suggestions.json", 'r') as f:
    suggestions = json.load(f)

stop_words = ['a', 'an', 'the', 'and', 'but', 'or', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'other', 'such', 'only', 'own', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now']


def clean_text(response):
  response = response.lower()
  response = re.sub('[^\w\s]', '', response)
  tokens = word_tokenize(response)
  tokens = [token for token in tokens if token not in stop_words]
  tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
  print(tokens)
  response = ' '.join(str(token) for token in tokens)
  return response

def analyseSentiment(resp):
    resp = [resp]
    with open('count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    resp = cv.transform(resp).toarray()
    y_pred = model.predict(resp)
    pred = y_pred[0]
    for suggestion in suggestions['suggestions']:
            if suggestion['label'] == 'positive':
                spos = random.choice(suggestion['response'])
            elif suggestion['label'] == 'negative':
                sneg = random.choice(suggestion['response'])
            else:
                sneu = random.choice(suggestion['response'])

    if pred==1:
        result = "Positive response"
        sugg = spos
    elif pred==-1:
        result = "Negative response"
        sugg = sneg
    else:
        result = "Neutral response"
        sugg = sneu

    return result, sugg

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/result', methods=['GET', 'POST'])
def result():
    txt = request.form.get("txt")
    print(txt)
    if request.method=='POST':
        if txt!="":
            res = clean_text(txt)
            print(res)
            result, sugg = analyseSentiment(res)
            return render_template("result.html", result = result, sugg = sugg, txt=txt)
        else:
            return render_template('home.html')
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True)