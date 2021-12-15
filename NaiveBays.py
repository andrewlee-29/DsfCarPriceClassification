import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
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


gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(Xtrain, Ytrain)

#Predict the response for test dataset
y_pred = gnb.predict(Xtest)

print("Naive Bays classification is ", accuracy_score(Ytest, y_pred)*100 )