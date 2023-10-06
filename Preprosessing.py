import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_X_y():
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
    return X, y

def get_train_test_vald():
    X, y = get_X_y()
    #splitting into testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state = 42)
    return X_train, y_train, X_test, y_test, X_val, y_val