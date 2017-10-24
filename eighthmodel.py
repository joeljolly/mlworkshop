from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(features,labels,test_size = .3)

clf = RandomForestClassifier()
clf.fit(Xtrain,Ytrain)

p=clf.predict(Xtest)

#print(Ytest)
#print(p)
from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(Ytest,p))
