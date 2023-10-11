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
from preprosessing import get_train_test_vald, transfrom_user_input

#Get features and labels
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_vald()


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

    #Store training errors:
    lin_train_errors = []
    lin_train_medians = []

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
        y_train_pred = lin_regr.predict(X_train_poly)
        y_pred = lin_regr.predict(X_val_poly)
        y_test_pred = lin_regr.predict(X_test_poly)

        #lasso implementation that is not currently used
        '''linlasso = Lasso(alpha=0.85, max_iter = 200000).fit(X_train_poly, y_train)
        y_pred = linlasso.predict(X_test_poly)'''

        #calculating error
        train_error = mean_squared_error(y_train, y_train_pred)
        train_median = np.median(np.absolute(y_train - y_train_pred))

        val_error = mean_squared_error(y_val, y_pred)
        val_median = np.median(np.absolute(y_val - y_pred))

        test_error = mean_squared_error(y_test, y_test_pred)
        test_median = np.median(np.absolute(y_test - y_test_pred))

        lin_val_errors.append(val_error)
        lin_val_medians.append(val_median)
        lin_test_errors.append(test_error)
        lin_test_medians.append(test_median)
        lin_train_errors.append(train_error)
        lin_train_medians.append(train_median)

        #Write the results for a later inspection
        f.write("Polynomial degree = {}".format(i))
        f.write("\nMean error: \n train: {} \n validation: {} \n".format(train_error, val_error))
        f.write("Median error: \n train: {} \n validation: {} \n\n".format(train_median, val_median))

    #Plottig the errors to a heatmap
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
    num_layers = range(int(input("\nStarting number of hidden layers?: \n")), int(input("Max amount of hidden layers?: \n")) + 1)    # number of hidden layers
    list_of_num_neurons = range(int(input("\nStarting number of neurons?: \n")), int(input("\nMax amount of neurons?:\n")) + 1, int(input("\nJump between neuron amounts?: \n"))) # number of neurons in each layer
    print("\n\nAmount of cycles: {}".format(len(list_of_num_neurons)))

    # Store errors for future plotting       
    mlp_val_errors = [[] for _ in list_of_num_neurons]
    mlp_test_errors = [[] for _ in list_of_num_neurons]

    mlp_val_medians = [[] for _ in list_of_num_neurons]
    mlp_test_medians = [[] for _ in list_of_num_neurons]
    for i, num_neurons in enumerate(list_of_num_neurons):
        print(str(i + 1))
        for j, num in enumerate(num_layers):
            #hidden_layer_sizes = tuple([num_neurons]*num) # size (num of neurons) of each layer stacked in a tuple
            hidden_layer_sizes = tuple([num_neurons]*num)
            
            
            # Mlp_regression conf
            mlp_regr = MLPRegressor()
            mlp_regr.hidden_layer_sizes = hidden_layer_sizes
            mlp_regr.max_iter = 100000
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
            f.write("number of layers|Neurons: {}|{}".format(num, num_neurons))
            f.write("\n\nMean error:\n")
            f.write(str(tr_error_mlp) + "\n")
            f.write(str(val_error_mlp))
            f.write("\n\nMedian error:\n")
            f.write(str(tr_median_mlp) + "\n")
            f.write(str(val_median_mlp))
            f.write("\n\n\n")

            #Keep track of every error type for later plotting
            mlp_val_errors[i].append(val_error_mlp)
            mlp_val_medians[i].append(val_median_mlp)

            mlp_test_errors[i].append(test_error_mlp)
            mlp_test_medians[i].append(test_median_mlp)

    #Reverse plots y axis
    mlp_val_errors.reverse()
    mlp_val_medians.reverse()

    #Plot the errors to a headmap
    fig, axes = plt.subplots(1, 2)
    fig.subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.4, hspace = 0.6)

    sns.heatmap(mlp_val_errors, annot=True, fmt='g', ax=axes[0], norm=LogNorm())
    axes[0].set_xlabel('layers',fontsize=8)
    axes[0].set_ylabel('Neurons/layer',fontsize=8)
    axes[0].set_title('Mean',fontsize=15)
    axes[0].xaxis.set_ticklabels(num_layers,fontsize=8)
    axes[0].yaxis.set_ticklabels(reversed(list_of_num_neurons),fontsize=8)

    sns.heatmap(mlp_val_medians, annot=True, fmt='g', ax=axes[1], norm=LogNorm())
    axes[1].set_xlabel('layers',fontsize=8)
    axes[1].set_ylabel('Neurons/layer',fontsize=8)
    axes[1].set_title('Median',fontsize=15)
    axes[1].xaxis.set_ticklabels(num_layers,fontsize=8)
    axes[1].yaxis.set_ticklabels(reversed(list_of_num_neurons),fontsize=8)

    plt.show()

    #Choose the best parameters and get test errors for those.
    inp1 = input("\nNeuron amount of your favorite cell?:\n")
    inp2 = input("\nLayer amount of your favorite cell?:\n")

    f.write("\n\nAll test means:\n")
    f.write(str(mlp_test_errors))
    f.write("\nAll test medians:\n")
    f.write(str(mlp_test_medians))
    f.write("\n\n\nChosen cell was neurons|layers: {}|{}".format(inp1, inp2))
    f.write("\n\nMean error:")
    f.write(str(np.sqrt(mlp_test_errors[list_of_num_neurons.index(int(inp1))][num_layers.index(int(inp2))])))
    f.write("\nMedian error:")
    f.write(str(mlp_test_medians[list_of_num_neurons.index(int(inp1))][num_layers.index(int(inp2))]))

    #Predict every vector from file "Own_predictions.txt" and print them out. 
    Own_prediction_mlp = MLPRegressor([int(inp1)]*int(inp2), max_iter= 100000, random_state=42).fit(X_train, y_train)

    for line in np.loadtxt("Own_prediction.txt", dtype=float):
        print(Own_prediction_mlp.predict(transfrom_user_input(np.array(line).reshape(1, -1))))

def main():
    choice = int(input("1 = Lin, 2 = Cnn: \n"))
    if choice == 1:
        lin()
    elif choice == 2:
        cnn()


if __name__ == "__main__":
    main()




