import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt

# Regression import 

from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, Lasso    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#data read and processing
df = pd.read_csv('Vuokraovi_21_9.csv')
df = df.assign(Sqm = pd.Series(map(lambda s: float(s.split()[0].replace(",", ".")), df['m^2'].astype(str))))
df = df.assign(PriceE = pd.Series(map(lambda s: float(s.split("€")[0].replace(",", ".").replace("\xa0", "")), df['Price'].astype(str))))
df = df.drop(['Place', 'Rooms', 'm^2', 'Price'], axis = 1)
data = df

features = []   # list for storing features of datapoints
labels = []     # list for storing labels of datapoints

#add all features and labels to list
for i in range(0, len(data)):     
    row = data.iloc[[i]]
    label = row['PriceE'].to_numpy()[0]
    row = row.drop('PriceE', axis = 1)
    feature = row.to_numpy().tolist()
    labels.append(label)
    features.append(feature)

#list reformatting
flat_list = []
for sublist in features:
    for item in sublist:
        flat_list.append(item)

#data into numpy
X = np.array(flat_list).reshape(len(data), len(data.columns) - 1) 
y = np.array(labels)

#scaling features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#splitting into testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)


degrees = [3]  # degrees to fit
    

for i in degrees:    #fit regression for chosen degrees
    
    print("Polynomial degree = ",i)

    #transform features into polynomial form
    poly = PolynomialFeatures(degree = i)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    #fit regression
    lin_regr = LinearRegression(fit_intercept=False)
    lin_regr.fit(X_train_poly, y_train)
    #predict testing values
    y_pred = lin_regr.predict(X_test_poly)

    #lasso implementation that is not currently used
    '''linlasso = Lasso(alpha=0.85, max_iter = 200000).fit(X_train_poly, y_train)
    y_pred = linlasso.predict(X_test_poly)'''

    #calculating error
    tr_error = mean_squared_error(y_test, y_pred)

    print("\ntraining error: \n",tr_error)
    print("\n\n")
 