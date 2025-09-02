from sklearn.ensemble import RandomForestClassifier
# Some number of decision trees. Each tree's root is the whole dataset, algorithm
# splits data based on features, trying to split as much as possible until they're
# classified into the most basic types. How basic we're getting is determined by
# some other metric, ex. something called Gini checks how probable is misclassification

classifier = RandomForestClassifier(random_state=0) # Forest of decision trees
# random_state=0 makes the rules more consistent. 0 is basically the seed, so the trees will
# be "built" similarly

x = [[ 1, 2, 3], # These are our two samples, each is a 3D vector? Basically each sample has
     [11,12,13]] # 3 features 
y = [0, 1] # The target values, we'll tell the model to associate one of the values with one of the
           # samples, and the other value with the other


classifier.fit(x,y) # We tell that [1,2,3] is of type "0", [11,12,13] is of type "1"


print(classifier.predict([[12,13,14],[22,23,24]])) # Predicting on new data after fitting the model
                                                   # Since they're both closer to [11,12,13], it
                                                   # predicts that they're both of type "1"