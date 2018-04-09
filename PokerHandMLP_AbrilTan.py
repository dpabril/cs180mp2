# Importing pertinent modules
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # function for shuffling dataframes
from sklearn.preprocessing import MinMaxScaler # object to be used for scaling/normalizing data
from sklearn.neural_network import MLPClassifier # The Multilayer Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Useful information for data analysis

# Loading the Poker Hand training and test data sets, subsequently shuffling their rows
train_set = pandas.read_csv('poker_train.csv')
train_set = shuffle(train_set)
print ("Training data set:\n", train_set)

test_set = pandas.read_csv('poker_test.csv')
test_set = shuffle(test_set)
print ("Test data set:\n", test_set)

# Preparations: The class column is dropped for both data sets
x_train = train_set.drop('class',axis=1) # class column dropped from the training set DataFrame
x_test = test_set.drop('class',axis=1) # class column dropped from the test set DataFrame

# Values in the class column for both data sets are stored in arrays
y_train = train_set['class'] # class column is isolated from the training set DataFrame
y_test = test_set['class'] # class column is isolated from the test set DataFrame

# Using sci-kit learn's <code>MinMaxScaler</code>, the data is scaled to the range [-1, 1] as preparation before being fed into the hyperbolic tangent activation function of the (yet-to-be-trained) multilayer perceptron.
scaler = MinMaxScaler(feature_range=(-1,1)) # setting the range [-1, 1] appropriate for the tanh activation function

x_train = scaler.fit_transform(x_train) # scaling the training set data to range [-1, 1]
x_test = scaler.transform(x_test) # scaling the test set data to range [-1, 1]

# Training the MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(10, 8),max_iter=100000,learning_rate_init=0.003,activation='tanh')
mlp.fit(x_train,y_train)

# Predictions are made using the recently trained MLP
train_pred = mlp.predict(x_train)
test_pred = mlp.predict(x_test)

# Deriving the confusion matrices of the training and test data sets
train_cnf = confusion_matrix(y_train, train_pred)
test_cnf = confusion_matrix(y_test,test_pred)

# Deriving classification reports from the predictions and target values
train_classreport = classification_report(y_train,train_pred)
test_classreport = classification_report(y_test,test_pred)

# Getting the accuracy scores
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

# Getting the confusion matrix and classification report for the training set
print ("Training set classification confusion matrix:\n")
print (train_cnf, "\n")

print ("Training set classification report:\n")
print (train_classreport)
print ("Training set accuracy:", train_acc, "\n\n")

# Getting the confusion matrix and classification report for the test set
print ("Test set classification confusion matrix:\n")
print (test_cnf, "\n")

print ("Test set classification report:\n")
print (test_classreport)
print ("Test set accuracy:", test_acc)


