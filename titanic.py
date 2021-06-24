import csv
import numpy as np
import pandas as pd

#reading the train.csv file into a pandas DataFrame:
train_table = pd.read_csv('train.csv')

#Looking at the table we have:
train_table.head()

# Also glance at the datatypes of each feature and the missing values
train_table.info()

# Only 3 columns have missing values: Age, Cabin and Embarked

# Cleaning the data as preparation for the analysis:

# Deleting Cabin as it has over 600 missing values
# Also deleting Ticket as it probably does not have any correlation with survival
train_table = train_table.drop(columns=['Cabin', 'Ticket'])

# now we can drop all the rows which contain any missing values
train_table = train_table.dropna()

# Review the distribution of the features
# First the categorical features:
train_table.describe(include=[object])

# The Name column have unique values in the dataset (total passangers are 712 after cleaning the dataset)
# Sex has two unique values (male, female), more male (top = male) as 63.6% (freq = 453 / count = 712) of the passangers being male
# Embarked takes on 3 values (C = Cherbourg, Q = Queenstown, S = Southampton), with Southampton being the port used by 77.8% passangers (freq = 554)

 # Then the numerical features
train_table.describe(include=[np.number])
# 40.4% of the passengers survived (mean = 0.404494)
# About 25%-25% of the passangers had 1st and 2nd class tickets and around 50% was travelling with 3rd class tickets (25th, 50th, and 75th percentiles)
# The passengers' age varied greatly between under 1 to 80, but half of the passenger were 28 or under
# The SibSp column contains the number of siblings/spouses, 
# and the Parch column the number of parents/children the passanger had aboard the Titanic
# The majority (around 75%) of passangers travelled with 1 or no relative.
# The ticket price went up to 512 USD, but 75% of the passangers paid less than 34 USD for their ticket

# Overviewing the survival rate by each feature
# Survived by sex in %
train_table[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survived by the location of embarkment in %
train_table[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survived by age in % -- age takes on too many values to have a good overview 
train_table[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survived by fare in % -- fare takes on too many values to have a good overview 
train_table[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survived by class in %
train_table[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survived by number of siblings/spouses in %
train_table[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survived by number of parents/childrens in %
train_table[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# writing cleaned train table into a csv file for Tableau visualisation
train_table.to_csv('train_cleaned.csv', index=False)

# creating one variable called single, as even though large families had the highest lowest rates, 
# most passenegers travelled withouth a family so it makes sense to include that in the model
train_table['Single'] = 0
train_table.loc[train_table['SibSp'] + train_table['Parch'] == 0, 'Single'] = 1

# converting categorical values into numeric:

# embarked
emb_val = {'C': 0, 'Q': 1, 'S': 2}
train_table['Embarked'] = train_table['Embarked'].map(emb_val)

# sex
sex_val = {'male': 0, 'female': 1}
train_table['Sex'] = train_table['Sex'].map(sex_val)

# rounding the floating point values 
train_table = train_table.round({'Fare': 1, 'Age': 1})


# creating bins for continuous values based on the precentiles
train_table[['Fare', 'Age']].describe(include=[np.number])

# fare
train_table.loc[train_table['Fare'] <= 8, 'Fare'] = 0
train_table.loc[(train_table['Fare'] > 8) & (train_table['Fare'] <= 15.65), 'Fare'] = 1
train_table.loc[(train_table['Fare'] > 15.65) & (train_table['Fare'] <= 33), 'Fare']   = 2
train_table.loc[train_table['Fare'] > 33, 'Fare'] = 3

# casting Fare values as int for the model:
train_table['Fare'] = train_table['Fare'].astype(int)

# checking the survival rate by the new Fare categories
train_table[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean()

# for Age based on the visualisation, it's preferable to create 5 bins rather than 4
train_table['Age_bin'] = pd.cut(train_table['Age'].astype(int), 5)

# checking the survival rate by the new Age categories
train_table[['Age_bin', 'Survived']].groupby(['Age_bin'], as_index=False).mean()

train_table.loc[ train_table['Age'] <= 16, 'Age'] = 0
train_table.loc[(train_table['Age'] > 16) & (train_table['Age'] <= 32), 'Age'] = 1
train_table.loc[(train_table['Age'] > 32) & (train_table['Age'] <= 48), 'Age'] = 2
train_table.loc[(train_table['Age'] > 48) & (train_table['Age'] <= 64), 'Age'] = 3
train_table.loc[(train_table['Age'] > 64), 'Age'] = 4

# dropping Name and Age bin as it is not needed for the model
train_table = train_table.drop(columns=['Name','Age_bin'])

# preparing test data for the model, same steps as with the train table
# reading the train.csv file into a pandas DataFrame:
test_table = pd.read_csv('test.csv')
# dropping unnecessary columns
test_table = test_table.drop(columns=['Cabin', 'Ticket', 'Name'])
# dropping all the rows which contain any missing values
test_table = test_table.dropna()

# converting categorical values into numeric:
test_table['Embarked'] = test_table['Embarked'].map(emb_val)
test_table['Sex'] = test_table['Sex'].map(sex_val)

# creating Single variable
test_table['Single'] = 0
test_table.loc[test_table['SibSp'] + test_table['Parch'] == 0, 'Single'] = 1

# rounding the floating values
test_table = test_table.round({'Fare': 1, 'Age': 1})

# creating Fare bins
test_table.loc[test_table['Fare'] <= 8, 'Fare'] = 0
test_table.loc[(test_table['Fare'] > 8) & (test_table['Fare'] <= 15.65), 'Fare'] = 1
test_table.loc[(test_table['Fare'] > 15.65) & (test_table['Fare'] <= 33), 'Fare']   = 2
test_table.loc[test_table['Fare'] > 33, 'Fare'] = 3

# casting Fare values as int for the model:
test_table['Fare'] = test_table['Fare'].astype(int)

# Age
test_table.loc[ test_table['Age'] <= 16, 'Age'] = 0
test_table.loc[(test_table['Age'] > 16) & (test_table['Age'] <= 32), 'Age'] = 1
test_table.loc[(test_table['Age'] > 32) & (test_table['Age'] <= 48), 'Age'] = 2
test_table.loc[(test_table['Age'] > 48) & (test_table['Age'] <= 64), 'Age'] = 3
test_table.loc[(test_table['Age'] > 64), 'Age'] = 4

# For the machin learning model I'm going to use the following independent variables or in other name features:
# features
x = train_table[['Sex', 'Pclass', 'Embarked', 'Fare', 'Age', 'Single']]

# And of course the dependent variable, the label is whether the person survived or not
# label
y = train_table['Survived']

# Creating the test set from the test table:
x_test = test_table[['Sex', 'Pclass', 'Embarked', 'Fare', 'Age', 'Single']]

# I'm going to use the two most common machine learning model Logistic regression and Random forest 
# and see which performs better

# Firstly, the regression:
# By the Wikipedia definition: logistic regression measures the relationship 
# between the categorical dependent variable and one or more independent variables 
# by estimating probabilities using a logistic function. Logistic regression is needed in our model
# as unlike ordinary linear regression, logistic regression is used for predicting dependent variables that take
# a limited number of categories rather than a continuous outcome
# (https://en.wikipedia.org/wiki/Logistic_regression)

# importing library for the model
from sklearn.linear_model import LogisticRegression

# Create a Classifier
clf = LogisticRegression()

# Train the model using the training data
clf.fit(x, y)

# predict for test data
y_pred_reg = clf.predict(x_test)

# checking accuracy of the model
accuracy_reg= round(clf.score(x, y) * 100, 2)
print("Accuracy: ", accuracy_reg)
# The mean training accuracy of the model is 79.21%

# Secondly, the random forest model:
# Random forests are an ensemble learning method for classification, regression 
# and other tasks that operates by constructing a multitude of decision trees at training time 
# (https://en.wikipedia.org/wiki/Random_forest)

# importing library for the model
from sklearn.ensemble import RandomForestClassifier

# Create a Classifier
clf=RandomForestClassifier(n_estimators=100)

# Train the model using the training data
clf.fit(x,y)

# predict for test data
y_pred_random=clf.predict(x_test)

# checking accuracy of the model
accuracy_random_forest = round(clf.score(x, y) * 100, 2)
print("Accuracy: ", accuracy_random_forest)
# The mean training accuracy of the model is 86.24%

# Since the random forest model perfomed way better, I'm saving that result:
output_random_forest = pd.DataFrame({'PassengerId': test_table.PassengerId, 'Survived': y_pred_random})
output_random_forest.to_csv('my_submission_random_forest.csv', index=False)




