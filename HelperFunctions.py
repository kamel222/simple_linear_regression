import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
class Helper:
    def Remove_My_String_Columns(my_data, my_columns):

        df = my_data.drop(my_columns, axis=1)

        """
        Remove columns with string values from pandas.DataFrame() object.

        This function should drop the columns with string values in the
        input `data` object. Dropping could be inplace or by returning
        a new pandas.DataFrame() object.

        Parameters
        ----------
        data : pandas.DataFrame()
            The dataframe that contains columns containing strings.
        columns : list
            A list containing the columns' names.

        Returns
        -------
        [optional] pandas.DataFrame()
            A copy of the cleaned dataframe.
        """
        #raise NotImplementedError("RemoveStringColumns() is not implemented yet!")

    def GetDateFeature(my_data, my_column):

        my_data[my_column] = pd.to_datetime(my_data[my_column]).values.astype(np.int64)
        df = my_data
        global df2
        df2=df

        """
        Returns a datetime component from a datetime feature.

        This function first

        the values into a datetime object. Then it returns one of the date time 
        components (e.g., Day, Month, or Year).

        Parameters
        ----------
        data: pandas.DataFrame()
            The data you have, it must contain at least the column with the issue
            and the column with the `Y` values.
        column: string
            The column name that contains date values.

        Returns
        -------
        numpy.ndarray()
            The cleaned feature containing a date component 
            (e.g., Day, Month, or Year).

        Note: Choose the date component you see fit.
        """
        #raise NotImplementedError("GetDateFeature() is not implemented yet!")


    def Linear_Regression_Model( feature, ground_truth):

        X = df2[feature]
        Y = df2[ground_truth]

        # ===========================================================#

        plt.scatter(X, Y)
        plt.xlabel('X_lable', fontsize=10)
        plt.ylabel('Y_lable', fontsize=10)
        plt.show()

        #===========================================================#

        cls = linear_model.LinearRegression()
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)

        # ===========================================================#

        cls.fit(X, Y)
        prediction = cls.predict(X)

        # ===========================================================#

        plt.scatter(X, Y)
        plt.xlabel('X_lable', fontsize=10)
        plt.ylabel('Y_lable', fontsize=10)
        plt.plot(X, prediction, color='green', linewidth=2)
        plt.show()

        # ===========================================================#

        print('\nMean Square Error : ', metrics.mean_squared_error(Y, prediction))

        # ===========================================================#


        FEATURE_Score = float(input('Enter your FEATURE score: '))
        x_test = np.array([FEATURE_Score])
        x_test = np.expand_dims(x_test, axis=1)
        y_test = cls.predict(x_test)
        print('\nYour predicted PRICE is : ' + str(float(y_test[0])))

        """
        Fit a Simple Linear Regression Model on the given feature.

        This function should train a Simple Linear Regression Model on 
        the given feature `X` and the ground truth values `Y`.

        Parameters
        ----------
        feature : numpy.ndarray()
            Array that contains the values for the desired input feature `X`.
        ground_truth : numpy.ndarray()
            Array that contains the actual `Y` values.

        Returns
        -------
        float
            The learned model parameter `C` or the `Y-intercept` of the line.
        float
            The learned model parameter `M` or the `slope` of the line.
        numpy.ndarray()
            A one-column array that contains your model predictions.
        """
        #raise NotImplementedError("LinearRegressionModel() is not implemented yet!")

