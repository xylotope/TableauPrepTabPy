# import libraries
import pandas
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump, load

# create function that converts 0, 1, 2 values to 0, 1
def cap(x: int) -> int:
  result = 1
  if x < 1:
    result = 0
  return result

# load the data and cap the result field
df = pandas.read_csv("./Data/diabetes_012_health_indicators_BRFSS2015.csv")
df['Diabetes_012'] = df['Diabetes_012'].apply(cap)

# grab the data into the independent and dependent variable fields
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
print(X)
print(y)

# split the training data from the testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train a logistic regression model
regr = linear_model.LogisticRegression()
regr.fit(X_train, y_train)

# predict the test values
y_pred = regr.predict(X_test)
print(y_pred)

# get a confusion matrix for the test results
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# get a report of the test results
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))

# dump the model to a file so it can be used later
dump(regr, 'regr.joblib')

# load the model back in and test it
regr2 = load('regr.joblib')
y_pred = regr2.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=target_names))
