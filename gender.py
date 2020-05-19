#Importing the libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

#Importing the dataset
df=pd.read_csv('NationalNames.csv')
df.Gender.replace({'F':0,'M':1},inplace=True)
df_data = df[["Name","Gender"]]
X=df_data['Name']
Y=df_data['Gender']

#Applying CountVectorizer
cv = CountVectorizer()
X1 = cv.fit_transform(X)
pickle.dump(cv, open("vector.pickel", "wb"))
pickle.dump(cv, open("vector.pickel", "wb"))

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size = 0.25, random_state = 0)

#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.5, dual=False, max_iter=10000, solver='lbfgs')
classifier.fit(X_train,Y_train)
classifier.score(X_test,Y_test)

filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(classification_report(Y_test, loaded_model.predict(X_test), digits=4))

"""#Making the predictions
Y_pred = classifier.predict(X_test)
data=['Barry']
vect=cv.transform(data)
p=classifier.predict(vect)
print(p)

print(classification_report(Y_test, classifier.predict(X_test), digits=4))
"""