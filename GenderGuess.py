from sklearn import tree

#test data that we are going to feed into our tree
#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37],
	 [166,65,40], [190,90,47], [175,75,42], [177,70,40],
	 [159,55,37], [171,75,42], [181,85,43]]

#correspoding genderst to given X values
Y = ['male', 'male', 'female', 'female', 'male','male',
	 'female', 'female','female','male','male']

#creats a tree that we are going to use on our dataset
clf = tree.DecisionTreeClassifier()
 
#Trains decision tree on data set
clf = clf.fit(X,Y)

#now we can predict whether we have a male or female by
#inputing a X value of [height, weight, shoe size]
prediction = clf.predict([[190,70,43]])

#now we can print our prediction 
print(prediction)