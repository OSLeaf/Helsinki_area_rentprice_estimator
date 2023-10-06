#Import nympy, matplotlib and seaborn libaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Import the needed functions from sklearn
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, Lasso    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error 
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

#Import the Preprosessing file
from Preprosessing import get_train_test_vald

#Get features and labels
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_vald()
f = open("Results.txt", "w")


def lin():
    '''
    Function to train LinearRegression model from the data
    '''
    #Open a file where to store the results
    f = open("LinearResults.txt", "w")

    #Store errors for plotting
    lin_val_errors = []
    lin_val_medians = []

    #Store errors for the final verdict
    lin_test_errors = []
    lin_test_medians = []

    degrees = range(0, int(input("Until which degree?: \n")) + 1)

    for i in degrees:    #fit regression for chosen degrees
        #transform features into polynomial form
        poly = PolynomialFeatures(degree = i)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.fit_transform(X_val)
        X_test_poly = poly.fit_transform(X_test)

        #fit regression
        lin_regr = LinearRegression(fit_intercept=False)
        lin_regr.fit(X_train_poly, y_train)
        #predict testing values
        y_pred = lin_regr.predict(X_val_poly)
        y_test_pred = lin_regr.predict(X_test_poly)

        #lasso implementation that is not currently used
        '''linlasso = Lasso(alpha=0.85, max_iter = 200000).fit(X_train_poly, y_train)
        y_pred = linlasso.predict(X_test_poly)'''

        #calculating error
        val_error = mean_squared_error(y_val, y_pred)
        val_median = np.median(np.absolute(y_val - y_pred))

        test_error = mean_squared_error(y_test, y_test_pred)
        test_median = np.median(np.absolute(y_test - y_test_pred))

        lin_val_errors.append(val_error)
        lin_val_medians.append(val_median)
        lin_test_errors.append(test_error)
        lin_test_medians.append(test_median)

        #Write the results for a later inspection
        f.write("Polynomial degree = {}".format(i))
        f.write("\nMean error: \n {} \n".format(val_error))
        f.write("Median error: \n {} \n\n".format(val_median))

    fig, axes= plt.subplots(2, 1)
    fig.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.6)

    sns.heatmap([np.asarray(lin_val_errors)], annot=True, fmt='g', ax=axes[0], norm=LogNorm())
    axes[0].set_xlabel('Degree',fontsize=8)
    axes[0].set_title('Mean',fontsize=15)

    sns.heatmap([np.asarray(lin_val_medians)], annot=True, fmt='g', ax=axes[1])
    axes[1].set_xlabel('Degree',fontsize=8)
    axes[1].set_title('Median',fontsize=15)
    plt.show()
    
    inp = int(input("Which degree is the best?: \n"))
    f.write("\n\n Chosen degree was = {}".format(inp))
    f.write("\n Mean error: {}".format(np.sqrt(lin_test_errors[inp])))
    f.write("\n Median error: {}".format(lin_test_medians[inp]))
    

def cnn():
    '''
    Function to train Mlp_Regression model from the data
    '''
    #Open a file where to store the results
    f = open("CnnResults.txt", "w")
    ## define a list of values for the number of hidden layers
    num_layers = range(1, int(input("Until how many hidden layers?: \n")) + 1)    # number of hidden layers
    num_neurons = int(input("\nHow many neurons per layer?: \n")) # number of neurons in each layer

    # Store errors for future plotting
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
        
        ## evaluate the trained MLP on training, validation and test sets
        y_pred_train_mlp = mlp_regr.predict(X_train)    # predict on the training set
        tr_error_mlp = mean_squared_error(y_train, y_pred_train_mlp)    # calculate the training error
        tr_median_mlp = np.median(np.absolute(y_train - y_pred_train_mlp))

        y_pred_val_mlp = mlp_regr.predict(X_val) # predict values for the validation data 
        val_error_mlp = mean_squared_error(y_val, y_pred_val_mlp) # calculate the validation error
        val_median_mlp = np.median(np.absolute(y_val - y_pred_val_mlp))

        y_pred_test_mlp = mlp_regr.predict(X_test) # predict values for the test data 
        test_error_mlp = mean_squared_error(y_test, y_pred_test_mlp) # calculate the test error
        test_median_mlp = np.median(np.absolute(y_test - y_pred_test_mlp))
        
        # Write the data to a file for later inspection
        f.write("number of layers: {}".format(num))
        f.write("\n\nMean error:\n")
        f.write(str(tr_error_mlp) + "\n")
        f.write(str(val_error_mlp))
        f.write("\n\nMedian error:\n")
        f.write(str(tr_median_mlp) + "\n")
        f.write(str(val_median_mlp))
        f.write("\n\n\n")

        #Keep track of every error type for later plotting
        mlp_tr_errors.append(tr_error_mlp)
        mlp_tr_median.append(tr_median_mlp)

        mlp_val_errors.append(val_error_mlp)
        mlp_val_median.append(val_median_mlp)

        mlp_test_errors.append(test_error_mlp)
        mlp_test_median.append(test_median_mlp)


    #Make a plot from the generated error values 
    figure, axis = plt.subplots(2)
    figure.tight_layout()

    axis[0].plot(num_layers, mlp_tr_errors, label = 'Train')
    axis[1].plot(num_layers, mlp_tr_median, label = 'Train')

    axis[0].plot(num_layers, mlp_val_errors,label = 'Valid')
    axis[1].plot(num_layers, mlp_val_median, label = 'Valid')
    
    axis[0].legend(loc="upper right")
    axis[1].legend(loc="upper right")

    axis[0].set_xticks(num_layers)
    axis[1].set_xticks(num_layers)
    axis[0].set_xlabel('Layers')
    axis[1].set_xlabel('Layers')
    axis[0].set_ylabel('Loss')
    axis[1].set_ylabel('Loss')
    axis[0].set_title('Train vs validation loss')
    axis[1].set_title('Train vs validation median')

    plt.show()


def main():
    choice = int(input("1 = Lin, 2 = Cnn: \n"))
    if choice == 1:
        lin()
    elif choice == 2:
        cnn()


if __name__ == "__main__":
    main()




