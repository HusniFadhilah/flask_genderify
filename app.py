from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/back')
def back():
	return render_template("index.html")

@app.route('/genderapi/<name>',methods=['GET'])
def predictApi(name):
    filename = 'finalized_model.sav'
    data = [name]
    cv = pickle.load(open("vector.pickel", 'rb'))     #Load vectorizer
    loaded_model = pickle.load(open(filename, 'rb'))
    vect=cv.transform(data).toarray()
    my_prediction=loaded_model.predict(vect)
    if(my_prediction==1):
        return jsonify({'Gender': 'Male'})
    return jsonify({'Gender': 'Female'})


@app.route('/predict',methods=['POST'])
def predict():
    filename = 'finalized_model.sav'
    cv = pickle.load(open("vector.pickel", 'rb'))
    classifier = pickle.load(open(filename, 'rb'))
    if request.method=='POST':
        nm=request.form['name']
        data=[nm]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html',prediction = my_prediction, name=nm)   
if __name__=='__main__':
    app.run(debug=True)