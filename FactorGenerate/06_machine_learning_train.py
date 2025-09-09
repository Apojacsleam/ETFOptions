import copy
import random
import joblib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import RESULT_PATH
from config import load_pickle, dump_pickle
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.linear_model import LassoLarsCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


def Lasso(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")
    model = LassoLarsCV(cv=5)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    predict_Rsquare = model.score(test_X, test_y)
    return y_pred, predict_Rsquare


def Ridge(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")
    model = RidgeCV(cv=5)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    predict_Rsquare = model.score(test_X, test_y)
    return y_pred, predict_Rsquare


def ElasticNet(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")
    model = ElasticNetCV(cv=5)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    predict_Rsquare = model.score(test_X, test_y)
    return y_pred, predict_Rsquare


def PCR(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Perform Principal Component Regression
    # Step 1: Perform PCA on training data
    pca = PCA()
    pca.fit(train_X)
    train_X_pca = pca.transform(train_X)

    # Step 2: Determine number of principal components to retain
    variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(variance_ratio_cumsum >= 0.95) + 1

    # Step 3: Reduce dimensionality of training and test data
    pca = PCA(n_components=n_components)
    pca.fit(train_X)
    train_X_pca = pca.transform(train_X)
    test_X_pca = pca.transform(test_X)

    # Step 4: Fit PCR model
    model = LinearRegression()
    model.fit(train_X_pca, train_y)

    # Step 5: Predict using the PCR model
    y_pred = model.predict(test_X_pca)

    # Step 6: Calculate R^2 for training and test data
    train_Rsquare = model.score(train_X_pca, train_y)
    predict_Rsquare = model.score(test_X_pca, test_y)

    return y_pred, predict_Rsquare


def PLS(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Fit Partial Least Squares (PLS) model
    model = PLSRegression(n_components=2)
    model.fit(train_X, train_y)

    # Predict using the PLS model
    y_pred = model.predict(test_X)
    y_pred = np.reshape(y_pred, (y_pred.shape[0],))
    # Calculate R^2 for training and test data
    predict_Rsquare = model.score(test_X, test_y)

    return y_pred, predict_Rsquare


def RF(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100)

    # Fit the model on training data
    rf_regressor.fit(train_X, train_y)

    # Predict on test data
    y_pred = rf_regressor.predict(test_X)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


def GBR(train_X, train_y, test_X, test_y):
    # Initialize the Gradient Boosting Regressor
    gbr_regressor = GradientBoostingRegressor(n_estimators=100)

    # Fit the model on training data
    gbr_regressor.fit(train_X, train_y)

    # Predict on test data
    y_pred = gbr_regressor.predict(test_X)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


def AdaBoost(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Initialize the AdaBoost Regressor
    ada_regressor = AdaBoostRegressor(n_estimators=100, random_state=42)

    # Fit the model on training data
    ada_regressor.fit(train_X, train_y)

    # Predict on test data
    y_pred = ada_regressor.predict(test_X)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


def GaussianRBM(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Initialize the GaussianRBM model
    rbm_model = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42)

    # Fit the model on training data
    rbm_model.fit(train_X, train_y)

    # Transform the data
    train_X_transformed = rbm_model.transform(train_X)
    test_X_transformed = rbm_model.transform(test_X)

    # Now you can use the transformed data with any other model, such as a regressor
    # For example, let's use Linear Regression

    # Initialize Linear Regression model
    lr_model = LinearRegression()

    # Fit the model on transformed training data
    lr_model.fit(train_X_transformed, train_y)

    # Predict on transformed test data
    y_pred = lr_model.predict(test_X_transformed)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


def MLP(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Initialize the MLP model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.01, activation='logistic')

    # Fit the model on training data
    mlp_model.fit(train_X, train_y)

    # Predict on test data
    y_pred = mlp_model.predict(test_X)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


def MLP2(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Initialize the MLP model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.01, activation='relu')

    # Fit the model on training data
    mlp_model.fit(train_X, train_y)

    # Predict on test data
    y_pred = mlp_model.predict(test_X)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


def MLP3(train_X, train_y, test_X, test_y):
    warnings.filterwarnings("ignore")

    # Initialize the MLP model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.01, activation='tanh')

    # Fit the model on training data
    mlp_model.fit(train_X, train_y)

    # Predict on test data
    y_pred = mlp_model.predict(test_X)

    # Calculate R^2 score
    r2 = r2_score(test_y, y_pred)

    return y_pred, r2


model_dict = {'Lasso': Lasso, 'Ridge': Ridge, 'ENet': ElasticNet, 'PCR': PCR, 'PLS': PLS, 'RF': RF, 'GaussianRBM': GaussianRBM, 'AdaBoost': AdaBoost, 'GBR': GBR, 'MLP': MLP,
              'MLP2': MLP2, 'MLP3': MLP3}


def process_train_split(data_dict: dict) -> list:
    warnings.filterwarnings('ignore')
    train_data = data_dict['train_data'].drop(columns=['option_code', 'month', 'etf']).fillna(0).convert_dtypes(convert_integer='int32').replace([np.inf, -np.inf], 0)
    predict_data = data_dict['predict_data'].reset_index(drop=True).fillna(0).convert_dtypes(convert_integer='int32').replace([np.inf, -np.inf], 0)

    for col in train_data.columns:
        if train_data[col].dtype == 'Float64' and col != 'next_return':
            maxv = train_data[col].max()
            minv = train_data[col].min()
            train_data[col] = (train_data[col] - minv) / (maxv - minv)
            predict_data[col] = (predict_data[col] - minv) / (maxv - minv)

    train_y = train_data['next_return'].copy(deep=True).values
    train_X = train_data.drop(columns=['next_return']).values

    predict_option_data = predict_data['option_code'].copy(deep=True)
    predict_month = predict_data['month'].copy(deep=True)
    predict_next_return = predict_data['next_return'].copy(deep=True)

    test_y = copy.deepcopy(predict_next_return).values
    test_X = predict_data.drop(columns=['option_code', 'month', 'next_return', 'etf']).values

    result_list = []
    for name, function in model_dict.items():
        y_pred, predict_Rsquare = function(train_X=copy.deepcopy(train_X), train_y=copy.deepcopy(train_y), test_X=copy.deepcopy(test_X), test_y=copy.deepcopy(test_y))
        result_df = pd.DataFrame({'option_code': copy.deepcopy(predict_option_data), 'month': copy.deepcopy(predict_month), 'next_return': copy.deepcopy(test_y),
                                  'predict_return': copy.deepcopy(y_pred)})
        result_list.append({'month': data_dict['month'], 'label': data_dict['label'], 'model': name, 'OS_Rsquare': predict_Rsquare, 'result_df': result_df})

    return result_list


if __name__ == '__main__':
    train_data_list = load_pickle(RESULT_PATH + '05_train_data_split2.pkl')

    random.shuffle(train_data_list)

    result_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_train_split)(data_dict) for data_dict in tqdm(train_data_list))

    dump_pickle(RESULT_PATH + '06_train_result.pkl', result_list)
