import numpy as np                   # import numpy package under shorthand "np"
import matplotlib.pyplot as plt

# Regression import 

from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, Lasso    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error 
from sklearn.neural_network import MLPRegressor

from Preprosessing import get_train_test_vald

#Get features and labels
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_vald()
f = open("Results.txt", "w")

def lin():
    degrees = range(0, int(input("Until which degree?: \n")) + 1)

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
        print("Median error:",np.median(np.absolute(y_test - y_pred)))
        print("\n\n")
    

def cnn():
    ## define a list of values for the number of hidden layers
    num_layers = range(1, int(input("Until how many hidden layers?: \n")))    # number of hidden layers
    num_neurons = int(input("\nHow many neurons per layer?: \n")) # number of neurons in each layer

    # we will use this variable to store the resulting training errors corresponding to different hidden-layer numbers
    mlp_tr_errors = []          
    mlp_val_errors = []
    mlp_test_errors = []

    mlp_tr_median = []
    mlp_val_median = []
    mlp_test_median = []

    for i, num in enumerate(num_layers):
        #hidden_layer_sizes = tuple([num_neurons]*num) # size (num of neurons) of each layer stacked in a tuple
        hidden_layer_sizes = tuple([num_neurons]*num)
        
        
        # Mlp_regression conf
        mlp_regr = MLPRegressor()
        mlp_regr.hidden_layer_sizes = hidden_layer_sizes
        mlp_regr.max_iter = 10000
        mlp_regr.random_state = 42
        
        mlp_regr.fit(X_train, y_train)
        
        ## evaluate the trained MLP on both training set and validation set
        y_pred_train_mlp = mlp_regr.predict(X_train)    # predict on the training set
        tr_error_mlp = mean_squared_error(y_train, y_pred_train_mlp)    # calculate the training error
        tr_median_mlp = np.median(np.absolute(y_train - y_pred_train_mlp))

        y_pred_val_mlp = mlp_regr.predict(X_val) # predict values for the validation data 
        val_error_mlp = mean_squared_error(y_val, y_pred_val_mlp) # calculate the validation error
        val_median_mlp = np.median(np.absolute(y_val - y_pred_val_mlp))

        y_pred_test_mlp = mlp_regr.predict(X_test) # predict values for the test data 
        test_error_mlp = mean_squared_error(y_test, y_pred_test_mlp) # calculate the test error
        test_median_mlp = np.median(np.absolute(y_test - y_pred_test_mlp))
        
        # sanity check num of layers
        # assert mlp_regr.n_layers_ == num_layers[i]+2 # total layers = num of hidden layers + input layer + output layer
        # # sanity check the error values
        # assert 3 < tr_error < 4 and 5 < val_error < 6
        f.write("number of layers: {}".format(num))
        f.write("\n\nMean error:\n")
        f.write(str(tr_error_mlp) + "\n")
        f.write(str(val_error_mlp))
        f.write("\n\nMedian error:\n")
        f.write(str(tr_median_mlp) + "\n")
        f.write(str(val_median_mlp))
        f.write("\n\n\n")

        mlp_tr_errors.append(tr_error_mlp)
        mlp_tr_median.append(tr_median_mlp)

        mlp_val_errors.append(val_error_mlp)
        mlp_val_median.append(val_median_mlp)

        mlp_test_errors.append(test_error_mlp)
        mlp_test_median.append(test_median_mlp)

    figure, axis = plt.subplots(2)
    figure.tight_layout()

    axis[0].plot(num_layers, mlp_tr_errors, label = 'Train')
    axis[1].plot(num_layers, mlp_tr_median, label = 'Train')

    axis[0].plot(num_layers, mlp_val_errors,label = 'Valid')
    axis[1].plot(num_layers, mlp_val_median, label = 'Train')

    axis[0].set_xticks(num_layers)
    axis[1].set_xticks(num_layers)
    axis[0].set_xlabel('Layers')
    axis[1].set_xlabel('Layers')
    axis[0].set_ylabel('Loss')
    axis[1].set_ylabel('Loss')
    axis[0].set_title('Train vs validation loss')
    axis[1].set_title('Train vs validation median')

    plt.legend(loc='upper center')
    plt.show()


def main():
    choice = int(input("1 = Lin, 2 = Cnn: \n"))

    if choice == 1:
        lin()
    elif choice == 2:
        cnn()


if __name__ == "__main__":
    main()




