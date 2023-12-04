import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# import all regression models from sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# import sklearn metrics, cross validation, grid search and pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
# import train test split from sklearn
from sklearn.model_selection import train_test_split
# import all scaling methods from sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.base import RegressorMixin

# import pearson correlation from scipy
from scipy.stats import pearsonr
from copy import deepcopy

from typing import Tuple, Any, List

class Regressor:
    """
    Class to make a regression model using sklearn. 
    I takes a dataframe and a target column and a list of features. 
    It splits the data into train and test and fits the model.
    It take a model from sklearn and fits it to the data. Additionally it can 
    use a scaler to scale the data and a gridsearch to find the best parameters.
    """
    def __init__(self, model, dataframe:pd.DataFrame,
                 #target is a list of strings
                target:list([str]),
                #features is a list of strings
                features:list([str]),
                #scaler is a sklearn scaler
                scaler:object=None):
        self.model = deepcopy(model)
        self.data = dataframe
        self.target = target
        self.features = features
        self.scaler = deepcopy(scaler)

        

    def train_test_split(self, splitter=None, test_size=0.2):
        """
        Splits the dataframe into train and test data.
        """
        if not splitter:
            self.data_train = self.data.iloc[:int(self.data.shape[0] * (1-test_size))]
            self.data_test = self.data.iloc[int(self.data.shape[0] * (1-test_size)):]
        else:
            self.data_train, self.data_test = splitter(self.data, test_size=test_size)
        self.X_train = self.data_train[self.features]
        self.y_train = self.data_train[self.target]
        self.X_test = self.data_test[self.features]
        self.y_test = self.data_test[self.target]
        return self.X_train, self.y_train, self.X_test, self.y_test

    def make_pipeline(self):
        """
        Makes a pipeline with the scaler and the model.
        """
        if self.scaler:
            self.pipeline = Pipeline([('scaler', self.scaler), ('model', self.model)])
        else:
            self.pipeline = Pipeline([('model', self.model)])
        return self.pipeline

    def fit(self):
        """
        Fits the model to the data.
        """
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def predict(self,data:pd.DataFrame) -> np.array:
        """
        Predicts the test data.
        """
        return self.pipeline.predict(data)

    def evaluate(self)->Tuple[float]:
        """
        Evaluates the model using r2, rmse and mae.
        
        Return:
        -------
        r2, rmse, mae, pearson
        """
        self.r2 = r2_score(self.y_test, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.pearson = pearsonr(self.y_test, self.y_pred)[0]
        return self.r2, self.rmse, self.mae, self.pearson

    def plot_results(self):
        # plot train and test data with the predicted values and another plot with the residuals
        fig, ax = plt.subplots(1,3, figsize=(12,6))
        sns.scatterplot(data=self.data_train, x=self.y_train, y=self.predict(self.X_train), ax=ax[0],)
        sns.scatterplot(data=self.data_test, x=self.y_test, y=self.y_pred, ax=ax[0])
        # add x=y line
        ax[0].plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'k--', lw=2)
        sns.scatterplot(data=self.data_test, x=self.y_test, y=self.y_pred-self.y_test, ax=ax[1], alpha=0.1)       
                # kde plot of the y_pred and y_test
        sns.kdeplot(data=self.y_pred, ax=ax[2], label='Predicted', fill=True)
        sns.kdeplot(data=self.y_test, ax=ax[2], label='True', fill=True)
        # legend
        ax[2].legend()
        # add a line at 0
        ax[1].axhline(0, color='black', linestyle='--')
        ax[0].set_xlabel('True')
        ax[0].set_ylabel('Predicted')
        ax[1].set_xlabel('True')
        ax[1].set_ylabel('Residual')
        ax[1].text(0.05, 0.95, f"RMSE: {self.rmse:.3f}\nMAE: {self.mae:.3f}", transform=ax[1].transAxes, verticalalignment='top')   
        ax[0].set_title(str(self.model).split("(")[0])
        ax[1].set_title(str(self.model).split("(")[0])
        ax[2].set_title(self.target)
        # add title to the whole figure
        # model name - scaler - target
        fig.suptitle(f"{str(self.model).split('(')[0]} - {str(self.scaler).split('(')[0]} - {self.target[0]}")
        plt.tight_layout()
        
    def plot_prediction(self, set:str='test'):
        """
        Plots the predicted vs the true values. and return the plot axis.
        """
        fig, ax = plt.subplots()
        if set == 'test':
            sns.scatterplot(data=self.data_test, x=self.y_test, y=self.y_pred, ax=ax)
        elif set == 'train':
            sns.scatterplot(data=self.data_train, x=self.y_train, y=self.y_pred, ax=ax)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(str(self.model).split("(")[0])
        return ax
        

    def print_equation(self):
        """
        Prints the equation of the model.
        """
        
        # print the regression equation
        eq = "y = "
        for i, feature in enumerate(self.features):
            eq += f"{self.model.coef_[i]:.6f}*{feature} + "
        eq += f" {self.model.intercept_:.6f}"
        print(f'{eq}')

    def fit_predict_evaluate(self):
        """
        Fits the model, predicts the test data and evaluates the model.
        """
        self.fit()
        self.y_pred = self.predict(self.X_test)
        self.evaluate()
        return self.r2, self.rmse, self.mae, self.pearson
    

def random_split(data:pd.DataFrame,
                test_size:float=0.2,
                random_state:int=42)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train and test data randomly.
    """
    data_train = data.sample(frac=1-test_size, random_state=random_state)
    data_test = data.drop(data_train.index)
    return data_train, data_test

def sabya_split(data:pd.DataFrame,
                test_size:float=0.2,
                random_state:int=42) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Splits the data into train and test data using train_test_split.
    """
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=random_state)    
    
    return data_train, data_test

def prepare_model(model:Regressor,
    data:pd.DataFrame, 
    target:str,
    features:List[str],
    scaler:StandardScaler,
    )->Tuple[Regressor, float, float, float, float]:
    """
    Prepare and evaluate a regression model.
    Args:
        model (Regressor): The regression model.
        data (DataFrame): The input data.
        target (str): The target variable.
        features (list): The list of feature variables.
        scaler (Scaler): The data scaler.

    Returns:
        tuple: A tuple containing the trained model (Regressor object),
            R-squared value (float), RMSE (float), MAE (float),
            and Pearson correlation coefficient (float).
    """

    # Copy the model and scaler
    model = deepcopy(model)
    scaler = deepcopy(scaler)

    # Create a Regressor instance
    m = Regressor(model, data, target, features,scaler=scaler)

    # Split the data into training and test sets
    m.train_test_split(splitter=random_split,test_size=0.2)

    # Create the pipeline
    m.make_pipeline()

    # Fit the model, predict, and evaluate the results
    r2, rmse, mae,pearsonr = m.fit_predict_evaluate()

    return m,r2,rmse,mae, pearsonr