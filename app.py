from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os

app=Flask(__name__)

port = int(os.environ.get('PORT', 5000))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/back')
def back():
	return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('NationalNames.csv')
    df.Gender.replace({'F':0,'M':1},inplace=True)
    df_data = df[["Name","Gender"]]
    X=df_data['Name']
    filename = 'finalized_model.sav'
    cv = CountVectorizer()
    X1=cv.fit_transform(X)
    classifier = pickle.load(open(filename, 'rb'))
    if request.method=='POST':
        nm=request.form['name']
        data=[nm]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html',prediction = my_prediction, name=nm)   
if __name__=='__main__':
   app.run(host='0.0.0.0', port=port, debug=True)