import argparse
import logging
import joblib
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)
def read_input(file_name, test_size=0.25):
    """Read input data and split it into train and test."""
    data = pd.read_csv(file_name)
    data.dropna(axis=0, subset=['SalePrice'], inplace=True)

    y = data.SalePrice
    X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

    train_X, test_X, train_y, test_y = train_test_split(X.values,
                                                      y.values,
                                                      test_size=test_size,
                                                      shuffle=False)

    imputer = SimpleImputer()
    train_X = imputer.fit_transform(train_X)
    test_X = imputer.transform(test_X)

    return (train_X, train_y), (test_X, test_y)

def train_model(train_X,
                train_y,
                test_X,
                test_y,
                n_estimators,
                learning_rate):
    """Train the model using XGBRegressor."""
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    model.fit(train_X,
            train_y,
            early_stopping_rounds=40,
            eval_set=[(test_X, test_y)])

    print("Best RMSE on eval: %.2f with %d rounds" %
               (model.best_score,
                model.best_iteration+1))
    return model

def eval_model(model, test_X, test_y):
    """Evaluate the model performance."""
    predictions = model.predict(test_X)
    logging.info("mean_absolute_error=%.2f", mean_absolute_error(predictions, test_y))

def save_model(model, model_file):
    """Save XGBoost model for serving."""
    joblib.dump(model, model_file)
    logging.info("Model export success: %s", model_file)
    
    
class nkTrain(object):
    
    def __init__(self):
        self.train_input = "data/train.csv"
        self.n_estimators = 50
        self.learning_rate = 0.1
        self.model_file = "trained_ames_model.dat"
        self.model = None

    def train(self):
        (train_X, train_y), (test_X, test_y) = read_input(self.train_input)
        model = train_model(train_X,
                          train_y,
                          test_X,
                          test_y,
                          self.n_estimators,
                          self.learning_rate)

        eval_model(model, test_X, test_y)
        save_model(model, self.model_file)

    def predict(self, X, feature_names=None):
        """Predict using the model for given ndarray."""
        if not self.model:
            self.model = joblib.load(self.model_file)
        # Do any preprocessing
        prediction = self.model.predict(data=X)
        # Do any postprocessing
        return prediction
model = nkTrain()
model.train()
