import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import sys


my_csv_file = sys.argv[1]
df=pd.read_csv(my_csv_file)


encoder = LabelEncoder()
cat_coll = ['Gender']
for cols in cat_coll:
    df[cols] = encoder.fit_transform(df[cols])

#print(df.head())

test_file = sys.argv[2]
test = pd.read_csv(test_file)
for cols in cat_coll:
    test[cols] = encoder.fit_transform(test[cols])
drop = test.drop("User ID",axis='columns')
#print(drop.head())

features = ['Gender','Age','EstimatedSalary','Purchased']
target =['Purchased']

X=df[features]
y=df[target]


tree = DecisionTreeClassifier()
tree = tree.fit(X,y)

predicted=tree.predict(drop)

#print(drop)
user = test['User ID']


#print(user)


ff=map(user,predicted)

import csv


f = open("demofile2.txt", "a")
for i in list(ff):
    f.write(f"{i} \n")
