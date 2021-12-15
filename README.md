# Data Scinece Foundation Project Car Price Prediction ( Economic Class Classification)


## Introduction
The objective of this project is to practice using python and different machine learnning techquiues to make some prediction. In this project, I am trying to predict the car price base on their features and characteristics. As a person who doesn't have professional knowledge about car, it is hard for us to determine is the car on the market price or not, espically when we try to purchase a used car.  I want to create an appliation solution that can tell me about is the car for low-middle economic class.

## Selection of Data
The data has over 10000 samples with 16 feautres: Make, Model, Year, Engine, Fuel, Type, Engine, HP,	Engine, Cylinders,	Transmission, Type,	Driven_Wheels,	Number of Doors,	Market, Category,	Vehicle, Size, Vehicle, Style, highway, MPG,	city, mpg,	Popularity,	MSRP. This is a perfect dataset for this project. It contains a lot of car feature, and also contain the MSRP( Manufacturer's Suggested Retail Price).

## Methods

Tools:
- NumPy, Pandas, and Scikit-learn for data analysis and inference
- GitHub for hosting/version control
- VS Code as IDE

Inference methods used with Scikit:
- Decision Tree model
- Random forest
- Features: train_test_split, LabelEncoder , accuracy_score

### Data cleaning and preparation

1. Since the result (mspr) is cont, I need to make it a discrete variable. I create another column called EcoClass and split the car into Low-middle(mspr<=60000) and High-middle(mspr>60000).

2. I delete the column (market category) for the model because they are difficult to encode and they don't really have a relationship between the price. I also delete the Engine Fuel Type because it is difficult to encode. I delete the MSRP because it is not the car feature.

3. I drop all null or nan data using the pandas.dropna .

4. Since there are String features in the dataset, I use LabelEncoder to encode the string feature to allow me the scikit modle able to read the feature.
### Data processing
1. I split the data as 75% training data and 25% testing data.
2. I use scikit-learn  Decision Tree model and random forest model.
## Results
(https://github.com/asdrewlee23/DsfCarPriceClassification/blob/main/DecisionTree.py)
(https://github.com/asdrewlee23/DsfCarPriceClassification/blob/main/Result.png)
## Discussion
As the
Without any limit(default setting), The decision tree model and random forest tree model results are around 97% which is close to 100%. It is definitely overfitting. To optimize the model, I tried to set the maximum tree depth. As the result, the tree is a sightly drop down to 95% when I set the tree depth to 2.

## Summary
This sample project deploys a supervised regression model to predict insurance costs based on 6 features. After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 73%. 

The web app is designed using Streamlit, and can do online and batch processing, and is deployed using Heroku and Streamlit. The Heroku app is live at https://ds-example.herokuapp.com/.
 
Streamlit is starting to offer free hosting as well. The same repo is also deployed at [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/memoatwit/dsexample/app.py)  
More info about st hosting is [here](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html).


## References
[1] [Car feauture dataset: kaggle](https://www.kaggle.com/CooperUnion/cardataset)