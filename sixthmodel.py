from scipy.spatial import distance
def eucli(a, b):
	return distance.euclidean(a,b)
class myKNN:
	#Fit method
	def fit(self, Xtrain,Ytrain):
		self.Xtrain=Xtrain
		self.Ytrain=Ytrain
	#Predict Method
	def predict(self,Xtest):
		predictions = []
		for row in Xtest:
			label = self.closest(row)
			predictions.append(label)
		return predictions
	#Closest
	def closest(self, row):
		best_dist = eucli(row, self.Xtrain[0])
		best_index =0
		for i in range(1,len(self.Xtrain)):
			distance =eucli(row, self.Xtrain[i])
			if(distance<best_dist):
				best_dist=distance
				best_index = i
		return self.Ytrain[best_index]

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(features,labels,test_size = .3)

clf = myKNN()
clf.fit(Xtrain,Ytrain)

p=clf.predict(Xtest)

#print(Ytest)
#print(p)
from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(Ytest,p))
