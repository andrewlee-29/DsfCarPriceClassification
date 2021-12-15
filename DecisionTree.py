import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
car_data =pd.read_csv("cardataset.csv")
# print(car_data)
## Drop the column
car_data.drop(['Model','Market Category','Engine Fuel Type', 'MSRP'], axis=1, inplace=True)
car_data= car_data.dropna()
## Encode the text label to number
le_make = LabelEncoder()
le_transmission_type = LabelEncoder()
le_driven_wheels = LabelEncoder()
le_vehicle_size = LabelEncoder()
le_vehicle_style = LabelEncoder()

car_data['Make_n']= le_make.fit_transform(car_data['Make'])
car_data['Transmission_type_n']= le_make.fit_transform(car_data['Transmission Type'])
car_data['Driven_Wheels_n']= le_make.fit_transform(car_data['Driven_Wheels'])
car_data['Vehicle Size_n']= le_make.fit_transform(car_data['Vehicle Size'])
car_data['Vehicle Style_n']= le_make.fit_transform(car_data['Vehicle Style'])
car_datan =car_data.drop(['Make','Transmission Type','Driven_Wheels','Vehicle Size','Vehicle Style'],axis='columns')
## Separte Target variable
X = car_datan.drop('EcoClass',axis="columns")
Y = car_data['EcoClass']
# print(X.head)
## Split data to Train and Test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=100)
##function to perform decision tree 
dtc= DecisionTreeClassifier(max_depth=2)
dtc.fit(Xtrain, Ytrain)
y_dtcpredict = dtc.predict(Xtest)

fig = plt.figure(figsize=(9, 11))
_ = tree.plot_tree(dtc,feature_names=list(X.columns))
fig.savefig("decision_tree.png", bbox_inches='tight')

print("Decision Tree with max depth 2 Accuracy is ", accuracy_score(Ytest, y_dtcpredict)*100 )

dtc= DecisionTreeClassifier(max_depth=3)
dtc.fit(Xtrain, Ytrain)
y_dtcpredict = dtc.predict(Xtest)
print("Decision Tree with max depth 3 Accuracy is ", accuracy_score(Ytest, y_dtcpredict)*100 )
dtc= DecisionTreeClassifier(max_depth=4)
dtc.fit(Xtrain, Ytrain)
y_dtcpredict = dtc.predict(Xtest)
print("Decision Tree with max depth 4 Accuracy is ", accuracy_score(Ytest, y_dtcpredict)*100 )
dtc= DecisionTreeClassifier(max_depth=5)
dtc.fit(Xtrain, Ytrain)
y_dtcpredict = dtc.predict(Xtest)
print("Decision Tree with max depth 5 Accuracy is ", accuracy_score(Ytest, y_dtcpredict)*100 )
dtc= DecisionTreeClassifier()
dtc.fit(Xtrain, Ytrain)
y_dtcpredict = dtc.predict(Xtest)
print("Decision Tree without max depth Accuracy is ", accuracy_score(Ytest, y_dtcpredict)*100 )
##function to perform random forest
rfc = RandomForestClassifier(max_depth=3)
rfc.fit(Xtrain, Ytrain)
y_rfcpredict = rfc.predict(Xtest)

print("Random Forest Tree Accuracy is ", accuracy_score(Ytest, y_rfcpredict)*100 )